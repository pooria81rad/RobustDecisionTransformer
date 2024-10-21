# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
from typing import Any, Dict, List, Optional, Tuple

import functions as func
import time
import copy
import os
import wandb
import traceback
import pyrallis
import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass
from tqdm import trange
from logger import init_logger, Logger
from attack import attack_dataset
from replay_buffer import ReplayBuffer
from networks import MLP

TensorBatch = List[torch.Tensor]

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


LOSS_FNS = {
    "A_L1": func.asymmetric_l1_loss,
    "A_L2": func.asymmetric_l2_loss,
    "L1": func.l1_loss,
    "L2": func.l2_loss,
}


@dataclass
class TrainConfig:
    # Experiment
    eval_every: int = 10
    eval_episodes: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 1000
    num_updates_on_epoch: int = 1000
    # model params
    n_hidden: int = 2
    hidden_dim: int = 256
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # IQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = True  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    q_loss_fn: str = "L2"  # L2, L1
    v_loss_fn: str = "A_L2"  # L2, L1, A_L2, A_L1
    # Wandb logging
    use_wandb: int = 0
    group: str = "2023082100"
    env: str = "halfcheetah-medium-v2"
    # env: str = "walker2d-medium-replay-v2"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    # dataset
    down_sample: bool = False
    sample_ratio: float = 1.0
    ######## others
    alg_type: str = os.path.basename(__file__).rstrip(".py")
    logdir: str = "~/results/corruption"
    dataset_path: str = "/apdcephfs/share_1563664/ztjiaweixu/datasets"
    save_model: bool = False
    ###### selection
    selection_agent: str = "none"
    threshold: float = 1.0
    ###### corruption
    corruption_agent: str = "IQL"
    corruption_seed: int = 2023
    corruption_mode: str = "none"  # none, random, adversarial
    corruption_obs: float = 1.0  # 0 or 1
    corruption_act: float = 0.0  # 0 or 1
    corruption_rew: float = 0.0  # 0 or 1
    corruption_next_obs: float = 0.0  # 0 or 1
    corruption_range: float = 0.1
    corruption_rate: float = 0.1
    use_original: int = 0  # 0 or 1
    same_index: int = 0
    froce_attack: int = 0

    def __post_init__(self):
        if self.env == "halfcheetah-medium-v2":
            if self.corruption_obs:
                self.threshold = 0.6
            if self.corruption_act:
                self.threshold = 1.6
        elif self.env == "walker2d-medium-replay-v2":
            if self.corruption_obs:
                self.threshold = 4.5
            if self.corruption_act:
                self.threshold = 2.5
        if self.env.startswith("antmaz"):
            self.eval_episodes = 100
            self.beta = 10.0
            self.iql_tau = 0.9
            self.normalize_reward = True
            self.buffer_size = 1000000
        key = self.env.split("-")[0]
        if key in ["door", "pen", "hammer", "relocate", "kitchen"]:
            self.beta = 0.5
            self.iql_tau = 0.7
            self.actor_dropout = 0.1
        # sample ratio
        if self.down_sample:
            if self.env.startswith("kitchen"):
                self.sample_ratio = 1.0
            if self.env.startswith("antmaze"):
                self.sample_ratio = 0.02
            if "medium-expert" in self.env:
                self.sample_ratio = 0.01
            if "medium-replay" in self.env:
                self.sample_ratio = 0.1
            if "medium-v2" in self.env:
                self.sample_ratio = 0.02
            key = self.env.split("-")[0]
            if key in ["door", "pen", "hammer", "relocate"]:
                self.sample_ratio = 0.01
        if self.corruption_mode == "random" and self.corruption_rew > 0.0:
            self.corruption_rew *= 30
        self.max_timesteps = self.num_epochs * self.num_updates_on_epoch


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP([state_dim, *([hidden_dim] * n_hidden), act_dim], dropout=dropout)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> MultivariateNormal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(
            self.max_action * action, -self.max_action, self.max_action
        )
        return action.cpu().data.numpy().flatten()

    def batch_act(self, state: np.ndarray, device: str = "cpu"):
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(
            self.max_action * action, -self.max_action, self.max_action
        )
        return action


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(
                self(state) * self.max_action, -self.max_action, self.max_action
            )
            .cpu()
            .data.numpy()
            .flatten()
        )

    def batch_act(self, state: np.ndarray, device: str = "cpu"):
        action = self(state)
        action = torch.clamp(
            self.max_action * action, -self.max_action, self.max_action
        )
        return action


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        q_loss_fn: str = "L2",
        v_loss_fn: str = "A_L2",
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau
        self.v_loss_fn = LOSS_FNS[v_loss_fn]
        self.q_loss_fn = LOSS_FNS[q_loss_fn]

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = self.v_loss_fn(adv, self.iql_tau).mean()
        log_dict["adv"] = adv.mean().item()
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v,
        observations,
        actions,
        rewards,
        terminals,
        log_dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(self.q_loss_fn(q - targets).mean() for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        func.soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(self, adv, observations, actions, log_dict):
        policy_out = self.actor(observations)
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


def train(config: TrainConfig, logger: Logger):
    # Set seeds
    func.set_seed(config.seed)

    if config.use_wandb:
        func.wandb_init(config)

    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if config.sample_ratio < 1.0:
        dataset_path = os.path.join(config.dataset_path, "original", f"{config.env}_ratio_{config.sample_ratio}.pt")
        dataset = torch.load(dataset_path)
    else:
        h5path = (
            config.dataset_path
            if config.dataset_path is None
            else os.path.expanduser(f"{config.dataset_path}/{config.env}.hdf5")
        )
        dataset = env.get_dataset(h5path=h5path)

    ##### corrupt
    if config.corruption_mode != "none":
        dataset, attack_indexs = attack_dataset(config, dataset, logger)

    dataset = d4rl.qlearning_dataset(env, dataset, terminate_on_end=True)

    dataset, state_mean, state_std = func.normalize_dataset(config, dataset)

    env = func.wrap_env(env, state_mean=state_mean, state_std=state_std)
    env.seed(config.seed)

    buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    buffer.load_d4rl_dataset(dataset)


    q_network = TwinQ(state_dim, action_dim, config.hidden_dim, config.n_hidden).to(
        config.device
    )
    v_network = ValueFunction(state_dim, config.hidden_dim, config.n_hidden).to(
        config.device
    )
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, config.hidden_dim, config.n_hidden, config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, config.hidden_dim, config.n_hidden, config.actor_dropout
        )
    ).to(config.device)
    logger.info(f"Actor Network: \n{str(actor)}")
    logger.info(f"Q Network: \n{str(q_network)}")
    logger.info(f"V Network: \n{str(v_network)}")

    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=3e-4)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=3e-4)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        "v_loss_fn": config.v_loss_fn,
        "q_loss_fn": config.q_loss_fn,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    logger.info("---------------------------------------")
    logger.info(f"Training IQL, Env: {config.env}, Seed: {config.seed}")
    logger.info("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    eval_log = func.eval(config, env, actor)
    logger.record("epoch", 0)
    for k, v in eval_log.items():
        logger.record(k, v)
    logger.dump(0)
    if config.use_wandb:
        wandb.log({"epoch": 0, **eval_log})

    best_reward = -np.inf
    total_updates = 0.0
    for epoch in trange(1, config.num_epochs + 1, desc="Training"):
        time_start = time.time()
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
            log_dict = trainer.train(batch)
            total_updates += 1
        time_end = time.time()
        epoch_time = time_end - time_start

        # Evaluate episode
        if epoch % config.eval_every == 0:  #  or epoch == config.num_epochs - 1:
            eval_log = func.eval(config, env, actor)
            logger.record("epoch", epoch)
            logger.record("epoch_time", epoch_time)
            for k, v in eval_log.items():
                logger.record(k, v)
            for k, v in log_dict.items():
                logger.record(f"update/{k}", v)
            logger.record("update/gradient_step", total_updates)
            logger.dump(epoch)

            if config.use_wandb:
                update_log = {f"update/{k}": v for k, v in log_dict.items()}
                wandb.log({"epoch": epoch, **update_log})
                wandb.log({"epoch": epoch, **eval_log})

            now_reward = eval_log["eval/reward_mean"]
            if now_reward > best_reward and config.save_model:
                best_reward = now_reward
                torch.save(
                    trainer.state_dict(),
                    os.path.join(logger.get_dir(), f"policy_best.pth"),
                )
                logger.info(
                    f"Save policy on epoch {epoch} for best reward {best_reward}."
                )

        if epoch % 500 == 0 and config.save_model:
            torch.save(
                trainer.state_dict(),
                os.path.join(logger.get_dir(), f"{epoch}.pt"),
            )
            logger.info(f"Save policy on epoch {epoch}.")

    if config.use_wandb:
        wandb.finish()


@pyrallis.wrap()
def main(config: TrainConfig):
    logger = init_logger(config)
    try:
        train(config, logger)
    except Exception:
        error_info = traceback.format_exc()
        logger.error(f"\n{error_info}")


if __name__ == "__main__":
    main()
