import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import Any, Dict, List, Optional, Tuple

import time
import math
import traceback
import json
import wandb
import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import utils.functions as func

from tqdm import trange
from torch.distributions import Normal
from copy import deepcopy
from dataclasses import dataclass
from utils.logger import init_logger, Logger
from utils.attack import attack_dataset
from utils.replay_buffer import ReplayBuffer
from utils.attack import Evaluation_Attacker

# general utils
TensorBatch = List[torch.Tensor]

@dataclass
class TrainConfig:
    # Wandb logging
    use_wandb: int = 0
    group: str = "2023082100"
    # model params
    n_hidden: int = 2
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    # training params
    buffer_size: int = 1_000_000
    group: str = "2024062701"
    env: str = "walker2d-medium-replay-v2"
    batch_size: int = 256
    num_epochs: int = 1000
    num_updates_on_epoch: int = 1000
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 1
    # general params
    checkpoints_path: Optional[str] = None
    seed: int = 0
    log_every: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    LCB_ratio: float = 6.0
    # evaluation params
    eval_id: str = "00"
    eval_only: bool = False
    eval_attack: bool = True
    eval_attack_eps: float = 0.01
    eval_attack_mode: str = "random"
    checkpoint_dir: str = None
    ######## others
    alg_type: str = os.path.basename(__file__).rstrip(".py")
    logdir: str = "~/results/corruption"
    dataset_path: str = "your_path_of_dataset"
    sample_ratio: float = 1.0
    ###### corruption
    corruption_agent: str = "IQL"
    corruption_seed: int = 2023
    corruption_mode: str = "adversarial"  # none, random, adversarial
    corruption_obs: float = 1.0  # 0 or 1
    corruption_act: float = 0.0  # 0 or 1
    corruption_rew: float = 0.0  # 0 or 1
    corruption_next_obs: float = 0.0  # 0 or 1
    corruption_range: float = 0.5
    corruption_rate: float = 0.1
    use_original: int = 0  # 0 or 1
    same_index: int = 1
    froce_attack: int = 1
    ######## UW
    use_UW: bool = True
    uncertainty_ratio: float = 0.7
    uncertainty_basic: float = 0.0
    uncertainty_min: float = 1
    uncertainty_max: float = 10

    def __post_init__(self):
        if self.corruption_mode == "random":
            if self.corruption_rew > 0:
                if self.env.startswith("halfcheetah"):
                    self.uncertainty_ratio = 0.7
                elif self.env.startswith("walker2d"):
                    self.uncertainty_ratio = 0.3
                elif self.env.startswith("hopper"):
                    self.uncertainty_ratio = 0.7
            elif self.corruption_next_obs > 0:
                if self.env.startswith("halfcheetah"):
                    self.uncertainty_ratio = 0.5
                elif self.env.startswith("walker2d"):
                    self.uncertainty_ratio = 0.5
                elif self.env.startswith("hopper"):
                    self.uncertainty_ratio = 0.7
        elif self.corruption_mode == "adversarial":
            if self.corruption_rew > 0:
                if self.env.startswith("halfcheetah"):
                    self.uncertainty_ratio = 0.7
                elif self.env.startswith("walker2d"):
                    self.uncertainty_ratio = 0.5
                elif self.env.startswith("hopper"):
                    self.uncertainty_ratio = 0.7
            elif self.corruption_next_obs > 0:
                if self.env.startswith("halfcheetah"):
                    self.uncertainty_ratio = 0.2
                elif self.env.startswith("walker2d"):
                    self.uncertainty_ratio = 0.5
                elif self.env.startswith("hopper"):
                    self.uncertainty_ratio = 1.0
        else:
            NotImplementedError
        # sample ratio
        if self.env.startswith("kitchen"):
            self.sample_ratio = 1.0
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
        # evaluation
        if self.eval_only:
            if self.corruption_obs > 0: corruption_tag = "obs"
            elif self.corruption_act > 0: corruption_tag = "act"
            elif self.corruption_rew > 0: corruption_tag = "rew"
            for file_name in os.listdir(os.path.join(self.logdir, self.group, self.env)):
                if f"{corruption_tag}_{self.seed}_" in file_name:
                    self.checkpoint_dir = os.path.join(self.logdir, self.group, self.env, file_name)
                    break
            with open(os.path.join(self.checkpoint_dir, "params.json"), "r") as f:
                config = json.load(f)
            unoverwritten_keys = ["eval_episodes", "checkpoint_dir"]
            for key, value in config.items():
                if key in self.__dict__ and not key.startswith("eval") and key not in unoverwritten_keys:
                    try:
                        value = eval(value)
                    except:
                        pass
                    self.__dict__[key] = value
                    print(f"Set {key} to {value}")
            self.eval_episodes = 100

# SAC Actor & Critic implementation
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_hidden: int = 2,
        max_action: float = 1.0,
    ):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden)]
        model = []
        for i in range(len(dims) - 1):
            model.append(nn.Linear(dims[i], dims[i + 1]))
            model.append(nn.ReLU())
        self.trunk = nn.Sequential(*model)

        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action


class VectorizedCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_hidden: int = 2,
        num_critics: int = 10,
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden)]
        model = []
        for i in range(len(dims) - 1):
            model.append(VectorizedLinear(dims[i], dims[i + 1], num_critics))
            model.append(nn.ReLU())
        model.append(VectorizedLinear(dims[-1], 1, num_critics))
        self.critic = nn.Sequential(*model)

        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        # [num_critics, batch_size, state_dim + action_dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(
            self.num_critics, dim=0
        )
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values


class Trainer:
    def __init__(
        self,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        critic: VectorizedCritic,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_learning_rate: float = 1e-4,
        LCB_ratio: float = 4.0,
        use_UW: bool = False,
        uncertainty_ratio: float = 1,
        uncertainty_basic: float = 1.0,
        uncertainty_min: float = 1,
        uncertainty_max: float = np.infty,
        device: str = "cpu",  # noqa
    ):
        self.device = device

        self.actor = actor
        self.critic = critic
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.tau = tau
        self.gamma = gamma
        self.LCB_ratio = LCB_ratio

        # uncertainty weight
        self.use_UW = use_UW
        self.uncertainty_ratio = uncertainty_ratio
        self.uncertainty_basic = uncertainty_basic
        self.uncertainty_min = uncertainty_min
        self.uncertainty_max = uncertainty_max
        self.uncertainty = torch.ones((1,1))

        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.alpha = self.log_alpha.exp().detach()

    def _alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, action_log_prob = self.actor(state, need_log_prob=True)

        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()

        return loss

    def _actor_loss(self, state: torch.Tensor, action_old: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        action, action_log_prob = self.actor(state, need_log_prob=True)
        q_value_dist = self.critic(state, action)
        assert q_value_dist.shape[0] == self.critic.num_critics
        q_value_min = q_value_dist.mean(0).view(1, -1) - self.LCB_ratio * q_value_dist.std(0).view(1, -1)
        # needed for logging
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -action_log_prob.mean().item()
        loss = (self.alpha * action_log_prob.view(1, -1) - q_value_min).mean()
        return loss, batch_entropy, q_value_std

    def _critic_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(
                next_state, need_log_prob=True
            )
            q_next = self.target_critic(next_state, next_action)
            q_next = q_next - self.alpha * next_action_log_prob.view(1,-1)
            q_target = reward.view(1,-1) + self.gamma * (1 - done.view(1,-1)) * q_next.detach()

        q_values = self.critic(state, action)
        # [ensemble_size, batch_size] - [1, batch_size]
        if self.use_UW:
            self.uncertainty = torch.clip(self.uncertainty_basic + self.uncertainty_ratio * q_values.std(dim=0).view(1,-1).detach(), self.uncertainty_min, self.uncertainty_max)
            loss = ((q_values - q_target) ** 2 / self.uncertainty).mean(dim=1).sum(dim=0)
        else:
            loss = ((q_values - q_target) ** 2).mean(dim=1).sum(dim=0)
        return loss

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, reward, next_state, done = [arr.to(self.device) for arr in batch]

        # Alpha update
        alpha_loss = self._alpha_loss(state)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        # Actor update
        actor_loss, actor_batch_entropy, q_policy_std = self._actor_loss(state, action)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        critic_loss = self._critic_loss(state, action, reward, next_state, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #  Target networks soft update
        with torch.no_grad():
            func.soft_update(self.target_critic, self.critic, tau=self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            # a ~ U[-max_action, max_action]
            max_action = self.actor.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(action)

            q_random_std = self.critic(state, random_actions).std(0).mean().item()

        update_info = {
            "alpha_loss": alpha_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": actor_batch_entropy,
            "alpha": self.alpha.item(),
            "q_policy_std": q_policy_std,
            "q_random_std": q_random_std,
        }
        return update_info

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "alpha_optim": self.alpha_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optim"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optim"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()


def train(config: TrainConfig, logger: Logger):
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
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=config.device,
    )
    buffer.load_d4rl_dataset(dataset)

    # Actor & Critic setup
    actor = Actor(state_dim, action_dim, config.hidden_dim, config.n_hidden, max_action).to(config.device)
    critic = VectorizedCritic(
        state_dim, action_dim, config.hidden_dim, config.n_hidden, config.num_critics
    ).to(config.device)
    logger.info(f"Actor Network: \n{str(actor)}")
    logger.info(f"Critic Network: \n{str(critic)}")

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate)

    trainer = Trainer(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        alpha_learning_rate=config.alpha_learning_rate,
        LCB_ratio=config.LCB_ratio,
        use_UW=config.use_UW,
        uncertainty_ratio=config.uncertainty_ratio,
        uncertainty_basic=config.uncertainty_basic,
        uncertainty_min=config.uncertainty_min,
        uncertainty_max=config.uncertainty_max,
        device=config.device,
    )
    # saving config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

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
            update_info = trainer.update(batch)
            total_updates += 1
        time_end = time.time()
        epoch_time = time_end - time_start

        # evaluation
        if epoch % config.eval_every == 0:  #  or epoch == config.num_epochs - 1:
            eval_log = func.eval(config, env, actor)
            logger.record("epoch", epoch)
            logger.record("epoch_time", epoch_time)
            for k, v in eval_log.items():
                logger.record(k, v)
            for k, v in update_info.items():
                logger.record(f"update/{k}", v)
            logger.record("update/gradient_step", total_updates)
            logger.dump(epoch)

            if config.use_wandb:
                update_log = {f"update/{k}": v for k, v in update_info.items()}
                wandb.log({"epoch": epoch, **update_log})
                wandb.log({"epoch": epoch, **eval_log})

            now_reward = eval_log["eval/reward_mean"]
            if now_reward > best_reward:
                best_reward = now_reward
                torch.save(
                    trainer.state_dict(),
                    os.path.join(logger.get_dir(), f"policy_best.pth"),
                )
                logger.info(
                    f"Save policy on epoch {epoch} for best reward {best_reward}."
                )

        if epoch % 500 == 0:
            torch.save(
                trainer.state_dict(),
                os.path.join(logger.get_dir(), f"{epoch}.pt"),
            )
            logger.info(f"Save policy on epoch {epoch}.")

    if config.use_wandb:
        wandb.finish()

def test(config: TrainConfig, logger: Logger):
    func.set_seed(config.seed)

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

    # Actor & Critic setup
    actor = Actor(state_dim, action_dim, config.hidden_dim, config.n_hidden, max_action).to(config.device)
    actor.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, "1000.pt"))["actor"])
    actor.eval()
    logger.info(f"Actor Network: \n{str(actor)}")

    attacker = None
    if config.eval_attack:
        state_std = func.get_state_std(config)
        attacker = Evaluation_Attacker(
            config.env, config.corruption_agent, config.eval_attack_eps,
            state_dim, action_dim, state_std, config.eval_attack_mode
        )

    eval_log = func.eval(config, env, actor, attacker)
    logger.record("epoch", 0)
    for k, v in eval_log.items():
        logger.record(k, v)
    logger.dump(0)


@pyrallis.wrap()
def main(config: TrainConfig):
    logger = init_logger(config)
    try:
        if config.eval_only:
            test(config, logger)
        else:
            train(config, logger)
    except Exception:
        error_info = traceback.format_exc()
        logger.error(f"\n{error_info}")


if __name__ == "__main__":
    main()
