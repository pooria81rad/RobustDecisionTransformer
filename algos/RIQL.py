# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import Any, Dict, List, Optional, Tuple, Union

import time
import json
import copy
import os
import wandb
import traceback
import pyrallis
import d4rl
import gym
import math
import numpy as np
import torch
import torch.nn as nn
import utils.functions as func

from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass
from tqdm import trange
from utils.logger import init_logger, Logger
from utils.attack import attack_dataset, Evaluation_Attacker
from utils.networks import MLP

TensorBatch = List[torch.Tensor]

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

@dataclass
class TrainConfig:
    # Experiment
    eval_every: int = 1
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
    # Wandb logging
    use_wandb: int = 0
    group: str = "2024062702"
    env: str = "walker2d-medium-replay-v2"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    flag: str = "test"
    sigma: float = 1.0
    num_actors: int = 1
    num_critics: int = 5
    quantile: float = 0.25
    # evaluation params
    eval_id: str = "00"
    eval_only: bool = False
    eval_attack: bool = True
    eval_attack_eps: float = 0.01
    eval_attack_mode: str = "random"
    checkpoint_dir: str = None
    # dataset
    down_sample: bool = True
    sample_ratio: float = 1.0
    ######## others
    alg_type: str = os.path.basename(__file__).rstrip(".py")
    logdir: str = "~/results/corruption"
    dataset_path: str = "your_path_of_dataset"
    save_model: bool = False
    ###### corruption
    corruption_agent: str = "IQL"
    corruption_seed: int = 2023
    corruption_mode: str = "none"  # none, random, adversarial
    corruption_obs: float = 0.0  # 0 or 1
    corruption_act: float = 0.0  # 0 or 1
    corruption_rew: float = 0.0  # 0 or 1
    corruption_next_obs: float = 0.0  # 0 or 1
    corruption_range: float = 0.1
    corruption_rate: float = 0.1
    use_original: int = 0  # 0 or 1
    same_index: int = 0
    froce_attack: int = 0

    def __post_init__(self):
        key = self.env.split("-")[0]
        if key in ["door", "pen", "hammer", "relocate", "kitchen"]:
            self.beta = 0.5
            self.iql_tau = 0.7
            self.actor_dropout = 0.1
        if self.env.startswith("antmaz"):
            self.eval_episodes = 100
            self.beta = 10.0
            self.iql_tau = 0.9
            self.normalize_reward = True
            self.buffer_size = 1000000
        if self.corruption_mode == "random":
            if self.corruption_obs:
                self.sigma = {
                    "walker2d": 0.1,
                    "hopper": 0.1,
                    "halfcheetah": 0.1,
                }.get(key, 1.0)
                self.quantile = {
                    "walker2d": 0.25,
                    "hopper": 0.25,
                    "halfcheetah": 0.1,
                }.get(key, 0.25)
                if key == "hopper":
                    self.num_critics = 3
            elif self.corruption_act:
                self.sigma = {
                    "walker2d": 0.5,
                    "hopper": 0.1,
                    "halfcheetah": 0.5,
                }.get(key, 1.0)
                self.quantile = {
                    "walker2d": 0.1,
                    "hopper": 0.25,
                    "halfcheetah": 0.25,
                }.get(key, 0.25)
                if key == "halfcheetah":
                    self.num_critics = 3
            elif self.corruption_rew:
                self.sigma = {
                    "walker2d": 3.0,
                    "hopper": 1.0,
                    "halfcheetah": 3.0,
                }.get(key, 1.0)
                self.quantile = {
                    "walker2d": 0.1,
                    "hopper": 0.25,
                    "halfcheetah": 0.25,
                }.get(key, 0.25)
        elif self.corruption_mode == "adversarial":
            if self.corruption_obs:
                self.sigma = {
                    "walker2d": 1.0,
                    "hopper": 1.0,
                    "halfcheetah": 0.1,
                }.get(key, 1.0)
                self.quantile = {
                    "walker2d": 0.25,
                    "hopper": 0.25,
                    "halfcheetah": 0.1,
                }.get(key, 0.25)
                if key == "hopper":
                    self.num_critics = 3
            elif self.corruption_act:
                self.sigma = {
                    "walker2d": 1.0,
                    "hopper": 1.0,
                    "halfcheetah": 1.0,
                }.get(key, 1.0)
                self.quantile = {
                    "walker2d": 0.1,
                    "hopper": 0.25,
                    "halfcheetah": 0.1,
                }.get(key, 0.25)
                if key == "halfcheetah":
                    self.num_critics = 3
            elif self.corruption_rew:
                self.sigma = {
                    "walker2d": 3.0,
                    "hopper": 0.1,
                    "halfcheetah": 1.0,
                }.get(key, 1.0)
                self.quantile = {
                    "walker2d": 0.1,
                    "hopper": 0.25,
                    "halfcheetah": 0.1,
                }.get(key, 0.25)
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
        # evaluation
        if self.eval_only:
            corruption_tag = ""
            if self.corruption_obs > 0: corruption_tag += "obs_"
            if self.corruption_act > 0: corruption_tag += "act_"
            if self.corruption_rew > 0: corruption_tag += "rew_"
            for file_name in os.listdir(os.path.join(self.logdir, self.group, self.env)):
                if f"{corruption_tag}{self.seed}_" in file_name:
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


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        attack_indexes: torch.Tensor,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device
        self._attack_indexes = torch.tensor(attack_indexes)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def sample_index(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        attack_indexes = self._attack_indexes[indices]
        return [states, actions, rewards, next_states, dones, attack_indexes]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def smooth_l1_loss(diff, sigma=1):
    beta = 1.0 / (sigma**2)
    diff = torch.abs(diff)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return loss


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


def asymmetric_l1_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * torch.abs(u))

class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(
            torch.empty(ensemble_size, in_features, out_features)
        )
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


class EnsemblePolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        num_actors: int = 3,
    ):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden)]
        model = []
        for i in range(len(dims) - 1):
            model.append(VectorizedLinear(dims[i], dims[i + 1], num_actors))
            model.append(nn.ReLU())
        model.append(VectorizedLinear(dims[-1], act_dim, num_actors))
        model.append(nn.Tanh())
        self.net = nn.Sequential(*model)
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(
                torch.mean(self(state), dim=0)[0] * self.max_action,
                -self.max_action,
                self.max_action,
            )
            .cpu()
            .data.numpy()
            .flatten()
        )


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


class VectorizedQ(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        num_critics: int = 5,
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
        # return torch.min(*self.both(state, action))
        res = self.both(state, action)
        return 0.75 * torch.min(*res) + 0.25 * torch.max(*res)


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
        sigma: float = 1.0,
        quantile: float = 0.25,
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
        self.sigma = sigma
        self.quantile = quantile

        self.total_it = 0
        self.device = device

    def _update_v(
        self, observations, actions, log_dict, attack_indexes
    ) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q_all = self.q_target(observations, actions)
            target_q = torch.quantile(target_q_all.detach(), self.quantile, dim=0)

            target_q_std = target_q_all.detach().std(dim=0)
            target_diff = target_q_all.detach().mean(dim=0) - target_q

        log_dict["attack_target_Q_std"] = torch.mean(
            target_q_std[torch.where(attack_indexes == 1)]
        ).item()
        log_dict["clean_target_Q_std"] = torch.mean(
            target_q_std[torch.where(attack_indexes == 0)]
        ).item()
        log_dict["attack_target_Q_diff"] = torch.mean(
            target_diff[torch.where(attack_indexes == 1)]
        ).item()
        log_dict["clean_target_Q_diff"] = torch.mean(
            target_diff[torch.where(attack_indexes == 0)]
        ).item()

        v = self.vf(observations)
        adv = target_q.detach() - v
        ######### average V
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)

        log_dict["value_loss"] = v_loss.item()
        log_dict["v_mean"] = torch.mean(v).item()
        log_dict["advantage_mean"] = torch.mean(adv).item()
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
        attack_indexes,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf(observations, actions)
        # q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        #################################### smooth l1 loss for Qs
        targets = torch.clamp(targets, -100, 1000).view(1, targets.shape[0])
        q_loss = smooth_l1_loss(targets.detach() - qs, sigma=self.sigma).mean()

        log_dict["q_loss"] = q_loss.item()
        log_dict["q_mean"] = torch.mean(qs).item()
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self, adv, observations, actions, next_observations, log_dict, attack_indexes
    ):
        batch_size, obs_dim = observations.shape[0], observations.shape[-1]
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=-1)
        else:
            raise NotImplementedError

        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_loss = exp_adv * bc_losses

        policy_loss = torch.mean(policy_loss)
        log_dict["actor_loss"] = policy_loss.item()
        log_dict["bc_loss"] = bc_losses.mean().item()
        log_dict["exp_weights"] = exp_adv.mean().item()
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        ############################ add clip norm
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
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
            attack_indexes,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict, attack_indexes)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(
            next_v, observations, actions, rewards, dones, log_dict, attack_indexes
        )
        # Update actor
        self._update_policy(
            adv, observations, actions, next_observations, log_dict, attack_indexes
        )

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


# @pyrallis.wrap()
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
    attack_indexes = np.zeros(dataset["rewards"].shape)
    if config.corruption_mode != "none":
        dataset, indexes = attack_dataset(config, dataset, logger)
        attack_indexes[indexes] = 1.0
    dataset = d4rl.qlearning_dataset(env, dataset, terminate_on_end=True)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(
            np.concatenate(
                [dataset["observations"], dataset["next_observations"]], axis=0
            ),
            eps=1e-3,
        )
    else:
        state_mean, state_std = 0, 1

    logger.info("state mean: ", state_mean)
    logger.info("state std: ", state_std)

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    env.seed(config.seed)

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        attack_indexes,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    # q_network = TwinQ(state_dim, action_dim).to(config.device)
    logger.info("num_critics: {}".format(config.num_critics))
    q_network = VectorizedQ(state_dim, action_dim, num_critics=config.num_critics).to(
        config.device
    )
    v_network = ValueFunction(state_dim).to(config.device)
    actor = DeterministicPolicy(
        state_dim, action_dim, max_action, config.hidden_dim, config.n_hidden, config.actor_dropout
    ).to(config.device)
    logger.info(f"Actor Network: \n{str(actor)}")
    logger.info(f"Q Network: \n{str(q_network)}")
    logger.info(f"V Network: \n{str(v_network)}")
    # actor = GaussianPolicy(state_dim, action_dim, max_action).to(config.device)
    # actor = EnsemblePolicy(state_dim, action_dim, max_action, num_actors=config.num_actors).to(config.device)

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
        "sigma": config.sigma,
        "quantile": config.quantile,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    logger.info("---------------------------------------")
    logger.info(f"Training RIQL, Env: {config.env}, Seed: {config.seed}")
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
            batch = replay_buffer.sample_index(config.batch_size)
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


def test(config: TrainConfig, logger: Logger):
    func.set_seed(config.seed)
    # Set seeds
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
    attack_indexes = np.zeros(dataset["rewards"].shape)
    if config.corruption_mode != "none":
        dataset, indexes = attack_dataset(config, dataset, logger)
        attack_indexes[indexes] = 1.0
    dataset = d4rl.qlearning_dataset(env, dataset, terminate_on_end=True)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(
            np.concatenate(
                [dataset["observations"], dataset["next_observations"]], axis=0
            ),
            eps=1e-3,
        )
    else:
        state_mean, state_std = 0.0, 1.0

    logger.info("state mean: ", state_mean)
    logger.info("state std: ", state_std)

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    env.seed(config.seed)

    actor = DeterministicPolicy(
        state_dim, action_dim, max_action, config.hidden_dim, config.n_hidden, config.actor_dropout
    ).to(config.device)
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
