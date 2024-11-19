# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import Optional, Tuple, Dict, List, Any, DefaultDict

import time
import json
import copy
import wandb
import traceback
import pyrallis
import d4rl
import gym
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils.functions as func
import utils.dt_functions as dt_func

from tqdm import trange
from dataclasses import dataclass
from collections import defaultdict
from utils.logger import init_logger, Logger
from utils.attack import attack_dataset, Evaluation_Attacker
from utils.networks import MLP

EXP_ADV_MAX = 100.0

@dataclass
class TrainConfig:
    # Experiment
    eval_every: int = 1
    eval_episodes: int = 10
    n_episodes: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 100
    num_updates_on_epoch: int = 1000
    # model params
    actor_net: str = "DT"  # MLP or DT
    # model params for MLP
    n_hidden: int = 2
    hidden_dim: int = 256
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # model params for DT
    embedding_dim: int = 128 # 768
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.0
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.0
    mlp_action: bool = False
    wo_return: bool = False  # Without return
    # IQL
    batch_size: int = 64  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = True  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    # DT
    reward_scale: float = 0.001
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    # Wandb logging
    use_wandb: int = 0
    group: str = "2024062702"
    env: str = "walker2d-medium-replay-v2"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    # hyper params
    sigma: float = 1.0
    num_actors: int = 1
    num_critics: int = 5
    quantile: float = 0.25
    wo_vfunc: bool = False  # Without value function
    lambda_q: float = 0.1
    LCB_ratio: float = 0.0
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
        # iql param
        key = self.env.split("-")[0]
        if key in ["door", "pen", "hammer", "relocate", "kitchen"]:
            self.beta = 0.5
            self.iql_tau = 0.7
            self.actor_dropout = 0.1
        if key == "antmaze":
            self.beta = 10.0
            self.iql_tau = 0.9
            self.eval_episodes = 100
            self.n_episodes = 100
            self.normalize_reward = True
        # target_returns and reward_scale
        if self.env.startswith("antmaze"):
            self.target_returns = [1.0, 0.5]
            self.reward_scale = 1.0
        if self.env.startswith("hopper"):
            self.target_returns = [3600, 1800]
            self.reward_scale = 0.001
        if self.env.startswith("halfcheetah"):
            self.target_returns = [12000, 6000]
            self.reward_scale = 0.001
        if self.env.startswith("walker"):
            self.target_returns = [5000, 2500]
            self.reward_scale = 0.001
        if self.env.startswith("kitchen"):
            self.target_returns = [400, 500]
            self.reward_scale = 1.0
        if self.env.startswith("door"):
            self.target_returns = [2900, 1450]
            self.reward_scale = 1.0
        if self.env.startswith("pen"):
            self.target_returns = [3100, 1550]
            self.reward_scale = 1.0
        if self.env.startswith("hammer"):
            self.target_returns = [12800, 6400]
            self.reward_scale = 1.0
        if self.env.startswith("relocate"):
            self.target_returns = [4300, 2150]
            self.reward_scale = 1.0
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
        if self.actor_net == "MLP":
            self.seq_len = 1
        self.max_timesteps = self.num_epochs * self.num_updates_on_epoch
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


def load_d4rl_trajectories(
    config: TrainConfig, logger: Logger = None
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    env_name = config.env
    if config.sample_ratio < 1.0:
        dataset_path = os.path.join(config.dataset_path, "original", f"{env_name}_ratio_{config.sample_ratio}.pt")
        dataset = torch.load(dataset_path)
    else:
        h5path = (
            config.dataset_path
            if config.dataset_path is None
            else os.path.expanduser(f"{config.dataset_path}/{env_name}.hdf5")
        )
        dataset = gym.make(env_name).get_dataset(h5path=h5path)

    attack_mask = np.ones_like(dataset["rewards"]) * -1
    if config.corruption_mode != "none":
        dataset, attack_indexs = attack_dataset(config, dataset, logger)
        attack_mask[attack_indexs] = 1
    dataset["attack_mask"] = attack_mask

    if config.normalize_reward:
        func.modify_reward(dataset, config.env)

    # dataset, state_mean, state_std = func.normalize_dataset(config, dataset)
    state_mean, state_std = 0.0, 1.0
    if config.normalize:
        state_mean = dataset["observations"].mean(0, keepdims=True).astype(np.float32)
        state_std = dataset["observations"].std(0, keepdims=True).astype(np.float32) + 1e-6

    min_return, max_return = np.inf, -np.inf
    traj, traj_len = [], []
    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])
        data_["dones"].append(dataset["terminals"][i])
        # data_["next_observations"].append(dataset["observations"][i + 1])
        data_["attack_mask"].append(dataset["attack_mask"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            if len(data_["rewards"]) > 1:
                episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
                # return-to-go if gamma=1.0, just discounted returns else
                episode_data["returns"] = dt_func.discounted_cumsum(
                    episode_data["rewards"], gamma=1.0
                )
                traj.append(episode_data)
                traj_len.append(episode_data["actions"].shape[0])
                min_return = min(min_return, episode_data["returns"].min())
                max_return = max(max_return, episode_data["returns"].max())
            # reset trajectory buffer
            data_ = defaultdict(list)
    logger.info(f"Min return: {min_return}, Max return: {max_return}")

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": state_mean, # dataset["observations"].mean(0, keepdims=True),
        "obs_std": state_std, # dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info


class Dataset:
    def __init__(self, config: TrainConfig, logger: Logger = None):
        self.dataset, info = load_d4rl_trajectories(config, logger)
        self.seq_len = config.seq_len + 1
        self.reward_scale = config.reward_scale
        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()
        self.device = config.device

    def correct(self, traj_indexs, time_steps, correct_data, correct_type):
        for i, (tarj_i, step_j) in enumerate(zip(traj_indexs, time_steps)):
            if step_j < self.dataset[tarj_i][correct_type].shape[0]:
                self.dataset[tarj_i][correct_type][step_j] = correct_data[i]

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len].reshape(-1, 1)
        time_steps = np.arange(start_idx, start_idx + self.seq_len)
        
        rewards = traj["rewards"][start_idx : start_idx + self.seq_len].reshape(-1, 1)
        dones = traj["dones"][start_idx : start_idx + self.seq_len].reshape(-1, 1)
        # next_states = traj["next_observations"][start_idx : start_idx + self.seq_len]

        attack_mask = traj["attack_mask"][start_idx : start_idx + self.seq_len].reshape(-1, 1)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale

        # next_states = (next_states - self.state_mean) / self.state_std
        rewards = rewards * self.reward_scale
        # pad up to seq_len if needed
        masks = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = dt_func.pad_along_axis(states, pad_to=self.seq_len)
            actions = dt_func.pad_along_axis(actions, pad_to=self.seq_len)
            returns = dt_func.pad_along_axis(returns, pad_to=self.seq_len)

            # next_states = dt_func.pad_along_axis(next_states, pad_to=self.seq_len)
            rewards = dt_func.pad_along_axis(rewards, pad_to=self.seq_len)
            dones = dt_func.pad_along_axis(dones, pad_to=self.seq_len, fill_value=1.0)

            attack_mask = dt_func.pad_along_axis(attack_mask, pad_to=self.seq_len)

        return states, actions, returns, rewards, dones, None, time_steps, masks

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        traj_ids = np.random.choice(np.arange(len(self.dataset)), size=batch_size, p=self.sample_prob, replace=True)

        state_list, action_list, return_list, reward_list, done_list, next_state_list, time_step_list, mask_list = \
            [], [], [], [], [], [], [], []
        for traj_id in traj_ids:
            start_idx = np.random.randint(0, self.dataset[traj_id]["rewards"].shape[0] - 1)
            states, actions, returns, rewards, dones, next_states, time_steps, masks = self.__prepare_sample(traj_id, start_idx)
            state_list.append(states)
            action_list.append(actions)
            return_list.append(returns)
            reward_list.append(rewards)
            done_list.append(dones)
            # next_state_list.append(next_states)
            time_step_list.append(time_steps)
            mask_list.append(masks)

        return [
            torch.tensor(state_list).to(self.device), torch.tensor(action_list).to(self.device), torch.tensor(return_list).to(self.device),
            torch.tensor(reward_list).to(self.device), torch.tensor(done_list).to(self.device), None,
            torch.tensor(time_step_list).to(self.device), torch.tensor(mask_list).to(self.device)
        ]


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


def set_actor(config: TrainConfig):
    if config.actor_net == "DT":
        actor = TransformerPlicy(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            max_action=config.max_action,
            embedding_dim=config.embedding_dim,
            seq_len=config.seq_len,
            episode_len=config.episode_len,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            attention_dropout=config.attention_dropout,
            residual_dropout=config.residual_dropout,
            embedding_dropout=config.embedding_dropout,
            mlp_action=config.mlp_action,
            wo_return=config.wo_return,
        ).to(config.device)
    elif config.actor_net == "MLP":
        actor = MLPPolicy(
            config.state_dim,
            config.action_dim,
            config.max_action,
            config.hidden_dim,
            config.n_hidden,
            config.actor_dropout
            ).to(config.device)
    return actor


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
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        if x.ndim == 4:
            # input: [ensemble_size, batch_size, squeeze, input_size]
            out = torch.einsum("eio,ebsi->ebso", self.weight, x) + self.bias.unsqueeze(1)
        elif x.ndim == 3:
            # input: [ensemble_size, batch_size, input_size]
            out = x @ self.weight + self.bias
        return out
    
    def extra_repr(self) -> str:
        return 'ensemble_size={}, in_features={}, out_features={}, bias={}'.format(
            self.ensemble_size, self.in_features, self.out_features, self.bias is not None
        )


class TransformerPlicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        mlp_action: bool = False,
        wo_return: bool = False,
    ):
        super().__init__()

        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                dt_func.TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        layer_num = 2 if mlp_action else 1
        self.action_head = dt_func.MLPBlock(embedding_dim, action_dim, layer_num, True)

        self.apply(self._init_weights)

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action
        self.predict_forward = False
        if wo_return:
            self.forward = self._forward_wo_return
        else:
            self.forward = self._forward

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states)
        act_emb = self.action_emb(actions)
        returns_emb = self.return_emb(returns_to_go)

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        sequence = sequence + time_emb.repeat_interleave(3, dim=1)
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        state_embed = out[:, 1::3]
        action_embed = out[:, 2::3]
        action_out = self.action_head(state_embed)
        return None, action_out, None

    def _forward_wo_return(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states)
        act_emb = self.action_emb(actions)

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * seq_len, self.embedding_dim)
        )
        sequence = sequence + time_emb.repeat_interleave(2, dim=1)
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 2 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        state_embed = out[:, 0::2]
        action_out = self.action_head(state_embed)
        return None, action_out, None


class MLPPolicy(nn.Module):
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


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class Trainer:
    def __init__(
        self,
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
        lambda_q: float = 0.1,
        LCB_ratio: float = 4.0,
        wo_vfunc: bool = False,
        device: str = "cpu",
    ):
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau
        self.sigma = sigma
        self.quantile = quantile
        self.lambda_q = lambda_q
        self.LCB_ratio = LCB_ratio
        self.wo_vfunc = wo_vfunc

        self.total_it = 0
        self.device = device

    def _update_v(
        self, states, actions, masks, log_dict
    ) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q_all = self.q_target(states, actions)
            target_q = torch.quantile(target_q_all.detach(), self.quantile, dim=0)

        v = self.vf(states)
        adv = target_q.detach() - v
        ######### average V
        v_loss = asymmetric_l2_loss(adv * masks, self.iql_tau)

        log_dict["value_loss"] = v_loss.item()
        log_dict["v_mean"] = torch.mean(v[masks.bool()]).item()
        log_dict["advantage_mean"] = torch.mean(adv[masks.bool()]).item()

        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self, next_v, states, actions, rewards, terminals, masks, log_dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf(states, actions)
        # q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        #################################### smooth l1 loss for Qs
        targets = torch.clamp(targets, -100, 1000)
        if len(targets.shape) < len(qs.shape):
            targets = targets.unsqueeze(0)
        else:
            targets = targets.min(dim=0, keepdim=True).values
        q_loss = smooth_l1_loss(targets.detach() - qs, sigma=self.sigma) # .mean()
        q_loss = (q_loss * masks.unsqueeze(0)).mean()

        qs_mask = torch.vstack([q[masks.bool()] for q in qs])
        log_dict["q_loss"] = q_loss.item()
        log_dict["q_mean"] = qs_mask.mean(0).mean().item()
        log_dict["q_std"] = qs_mask.std(0).mean().item()

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        func.soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self, adv, states, actions, returns, time_steps, masks, log_dict
    ):
        if isinstance(self.actor, MLPPolicy):
            policy_out = self.actor(states)
        else:
            padding_masks = ~masks.to(torch.bool)
            predicted = self.actor(
                states=states,
                actions=actions,
                returns_to_go=returns,
                time_steps=time_steps,
                padding_mask=padding_masks,
            )
            policy_out = predicted[1]
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=-1)
        else:
            raise NotImplementedError

        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_loss = exp_adv * bc_losses
        policy_loss = (policy_loss * masks).mean()

        log_dict["actor_loss"] = policy_loss.item()
        log_dict["bc_loss"] = bc_losses[masks.bool()].mean().item()
        log_dict["exp_weights"] = exp_adv[masks.bool()].mean().item()

        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        ############################ add clip norm
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def _update_policy_v2(
        self, states, actions, returns, time_steps, masks, log_dict
    ):
        if isinstance(self.actor, MLPPolicy):
            policy_out = self.actor(states)
        else:
            padding_masks = ~masks.to(torch.bool)
            predicted = self.actor(
                states=states,
                actions=actions,
                returns_to_go=returns,
                time_steps=time_steps,
                padding_mask=padding_masks,
            )
            policy_out = predicted[1]

        bc_loss = torch.sum((policy_out - actions) ** 2, dim=-1)
        bc_loss = bc_loss * masks

        q_value = self.qf(states, policy_out)
        q_value_min = q_value.mean(0) - self.LCB_ratio * q_value.std(0)
        q_value_min = q_value_min * masks

        policy_loss = self.lambda_q * torch.mean(-q_value_min) + torch.mean(bc_loss)

        q_value_mask = torch.vstack([q[masks.bool()] for q in q_value])
        log_dict["actor_loss"] = policy_loss.item()
        log_dict["bc_loss"] = bc_loss.mean().item()
        log_dict["q_loss"] = q_value.mean().item()
        log_dict["q_mean"] = q_value_mask.mean(0).mean().item()
        log_dict["q_std"] = q_value_mask.std(0).mean().item()

        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train_normal(self, batch: List[torch.Tensor]) -> Dict[str, float]:
        log_dict = {}

        states, actions, returns, rewards, dones, _, time_steps, masks = batch
        next_states = states[:, 1:]
        states = states[:, :-1]
        actions = actions[:, :-1]
        returns = returns[:, :-1]
        rewards = rewards[:, :-1]
        dones = dones[:, :-1]
        time_steps = time_steps[:, :-1]
        masks = masks[:, :-1]

        with torch.no_grad():
            next_v = self.vf(next_states)
        # Update value function
        adv = self._update_v(states, actions, masks, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(
            next_v, states, actions, rewards, dones, masks, log_dict
        )
        # Update actor
        self._update_policy(
            adv, states, actions, returns, time_steps, masks, log_dict
        )

        return log_dict

    def train_wo_vfunc(self, batch: List[torch.Tensor]) -> Dict[str, float]:
        log_dict = {}
        states, actions, returns, rewards, dones, _, time_steps, masks = batch

        with torch.no_grad():
            if isinstance(self.actor, MLPPolicy):
                next_actions = self.actor(states[:, 1:])
            else:
                padding_masks = ~masks.to(torch.bool)
                predicted = self.actor(
                    states=states[:, 1:],
                    actions=actions[:, 1:],
                    returns_to_go=returns[:, 1:],
                    time_steps=time_steps[:, 1:],
                    padding_mask=padding_masks[:, 1:],
                )
                next_actions = predicted[1]
            next_states = states[:, 1:]
            next_q = self.q_target(next_states, next_actions)

        states = states[:, :-1]
        actions = actions[:, :-1]
        returns = returns[:, :-1]
        rewards = rewards[:, :-1]
        dones = dones[:, :-1]
        time_steps = time_steps[:, :-1]
        masks = masks[:, :-1]

        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_q, states, actions, rewards, dones, masks, log_dict)
        # Update actor
        self._update_policy_v2(states, actions, returns, time_steps, masks, log_dict)

        return log_dict

    def train(self, batch: List[torch.Tensor]) -> Dict[str, float]:
        self.total_it += 1
        if self.wo_vfunc:
            return self.train_wo_vfunc(batch)
        else:
            return self.train_normal(batch)

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

    # dataset setup
    dataset = Dataset(config, logger)
    logger.info(f"Dataset: {len(dataset.dataset)} trajectories")
    logger.info(f"State mean: {dataset.state_mean}, std: {dataset.state_std}")

    # environment setup
    env = gym.make(config.env)
    env = func.wrap_env(
        env,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    env.seed(config.seed)
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]
    config.max_action = float(env.action_space.high[0])

    v_network = ValueFunction(config.state_dim, config.hidden_dim, config.n_hidden).to(config.device)
    q_network = VectorizedQ(
        config.state_dim, config.action_dim, config.hidden_dim, config.n_hidden, config.num_critics
    ).to(config.device)
    actor = set_actor(config)
    logger.info(f"Actor Network: \n{str(actor)}")
    logger.info(f"Q Network: \n{str(q_network)}")
    logger.info(f"V Network: \n{str(v_network)}")

    v_optimizer = optim.Adam(v_network.parameters(), lr=3e-4)
    q_optimizer = optim.Adam(q_network.parameters(), lr=3e-4)
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
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
        "lambda_q": config.lambda_q,
        "LCB_ratio": config.LCB_ratio,
        "wo_vfunc": config.wo_vfunc,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    logger.info("---------------------------------------")
    logger.info(f"Training {config.alg_type}, Env: {config.env}, Seed: {config.seed}")
    logger.info("---------------------------------------")

    # Initialize actor
    trainer = Trainer(**kwargs)

    if config.actor_net == "MLP":
        eval_log = func.eval(config, env, actor)
    elif config.actor_net == "DT":
        eval_log = dt_func.eval_fn(config, env, actor)
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
            batch = dataset.get_batch(config.batch_size)
            log_dict = trainer.train(batch)
            total_updates += 1
        time_end = time.time()
        epoch_time = time_end - time_start

        # Evaluate episode
        if epoch % config.eval_every == 0:  #  or epoch == config.num_epochs - 1:
            if config.actor_net == "MLP":
                eval_log = func.eval(config, env, actor)
            elif config.actor_net == "DT":
                eval_log = dt_func.eval_fn(config, env, actor)
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

            if config.actor_net == "MLP":
                now_reward = eval_log["eval/reward_mean"]
            elif config.actor_net == "DT":
                now_reward = eval_log[f"eval/{config.target_returns[0]}_reward_mean"]
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
        func.modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = func.compute_mean_std(
            np.concatenate(
                [dataset["observations"], dataset["next_observations"]], axis=0
            ),
            eps=1e-3,
        )
    else:
        state_mean, state_std = 0.0, 1.0

    logger.info("state mean: ", state_mean)
    logger.info("state std: ", state_std)

    dataset["observations"] = func.normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = func.normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    env = func.wrap_env(env, state_mean=state_mean, state_std=state_std)
    env.seed(config.seed)

    actor = MLPPolicy(state_dim, action_dim, max_action).to(config.device)
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
