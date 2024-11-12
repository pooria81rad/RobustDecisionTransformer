# inspiration:
# 1. https://github.com/hukz18/DeFog  # noqa
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import traceback
import json
import time
import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import torch
import wandb
import math
import utils.functions as func

from torch import nn
from torch.nn import functional as F
from tqdm.auto import trange  # noqa
from collections import defaultdict
from dataclasses import dataclass
from utils.networks import TransformerBlock
from utils.logger import init_logger, Logger
from utils.attack import attack_dataset
from utils.drop_fn import get_drop_fn
from utils.attack import Evaluation_Attacker


@dataclass
class TrainConfig:
    # Experiment
    eval_every: int = 1 # Log every n epochs
    n_episodes: int = 10  # How many episodes run during evaluation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    deterministic_torch: bool = False
    num_epochs: int = 100
    num_updates_on_epoch: int = 1000
    # model params
    embedding_dim: int = 128 # 768
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    model_drop_p: float = 0.1
    # training params
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    update_steps: int = 100_000
    warmup_steps: int = 10_000
    reward_scale: float = 0.001
    num_workers: int = 4
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    recalculate_return: bool = False
    # buffer:
    drop_fn: str = "const"
    drop_p: float = 0.8
    finetune_drop_p: float = 0.8
    update_interval: int = 500
    drop_aware: bool = True
    sample_type: str = "traj_length"
    # finetune
    train_steps: int = 50000
    finetune_steps: int = 10000
    # evaluation params
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    # Wandb logging
    use_wandb: int = 0
    group: str = "2023082100"
    env: str = "walker2d-medium-replay-v2"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    # evaluation params
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    eval_id: str = "00"
    eval_only: bool = False
    eval_attack: bool = True
    eval_attack_eps: float = 0.01
    eval_attack_mode: str = "random"
    checkpoint_dir: str = None
    ######## others
    debug: bool = False
    alg_type: str = os.path.basename(__file__).rstrip(".py")
    logdir: str = "~/results/corruption"
    dataset_path: str = "your_path_of_dataset"
    sample_ratio: float = 1.0
    ###### corruption
    corruption_agent: str = "EDAC"
    corruption_seed: int = 2023
    corruption_mode: str = "none"  # none, random, adversarial
    corruption_obs: float = 0.0  # 0 or 1
    corruption_act: float = 1.0  # 0 or 1
    corruption_rew: float = 0.0  # 0 or 1
    corruption_next_obs: float = 0.0  # 0 or 1
    # corruption_range: float = 0.1
    corruption_rate: float = 0.3
    use_original: int = 0  # 0 or 1
    same_index: int = 0
    froce_attack: int = 0

    def __post_init__(self):
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
        # finetune drop rate
        if "medium-expert-v2" in self.env:
            if "halfcheetah" in self.env:
                self.finetune_drop_p = 0.5
            else:
                self.finetune_drop_p = 0.9
        # others
        if self.env.startswith("antmaze"):
            self.num_epochs = 100
            self.n_episodes = 100
        if self.corruption_mode == "random" and self.corruption_rew > 0.0:
            self.corruption_rew *= 30
            self.recalculate_return = True
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
            unoverwritten_keys = ["n_episodes", "checkpoint_dir"]
            for key, value in config.items():
                if key in self.__dict__ and not key.startswith("eval") and key not in unoverwritten_keys:
                    try:
                        value = eval(value)
                    except:
                        pass
                    self.__dict__[key] = value
                    print(f"Set {key} to {value}")
            self.n_episodes = 100


# some utils functionalities specific for Decision Transformer
def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def load_d4rl_trajectories(
    config: TrainConfig, env_name: str, gamma: float = 1.0, logger: Logger = None
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
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

    # dataset, state_mean, state_std = func.normalize_dataset(config, dataset)
    state_mean, state_std = 0.0, 1.0
    if config.normalize:
        state_mean = dataset["observations"].mean(0, keepdims=True)
        state_std = dataset["observations"].std(0, keepdims=True) + 1e-6

    traj, traj_len = [], []

    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])
        data_["attack_mask"].append(dataset["attack_mask"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            # reset trajectory buffer
            data_ = defaultdict(list)

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": state_mean, # dataset["observations"].mean(0, keepdims=True),
        "obs_std": state_std, # dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info


class SequenceBuffer():
    # def __init__(self, config, dataset, context_len, gamma, sample_type='traj_length', seed=0) -> None:
    def __init__(self, config: TrainConfig, logger: Logger = None):
        dataset, info = load_d4rl_trajectories(config, config.env, gamma=1.0, logger=logger)
        self.num_trajs = len(dataset)

        self.state_dim = dataset[0]['observations'].shape[1]
        self.action_dim = dataset[0]['actions'].shape[1]
        self.context_len = config.seq_len
        self.size = sum([len(traj['observations']) for traj in dataset]) + 1  # plus one for padding zeros

        self.states = np.zeros((self.size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.size, self.action_dim), dtype=np.float32)
        self.rewards_to_go = np.zeros((self.size,), dtype=np.float32)

        self.device = config.device
        self.traj_length = np.zeros(self.num_trajs, dtype=np.int32)
        self.traj_sp = np.zeros(self.num_trajs, dtype=np.int32)  # trajectory start point
        self.traj_returns = np.zeros(self.num_trajs, dtype=np.float32)
        self.rng = np.random.default_rng(config.seed)
        traj_pointer = 0

        for i, traj in enumerate(dataset):
            self.traj_sp[i] = traj_pointer
            observations, actions, rewards = traj['observations'], traj['actions'], traj['rewards']
            # observations, actions, rewards = traj['observations'], traj['actions'], -traj['rewards']
            # observations, actions, rewards = traj['observations'], traj['actions'], self.rng.uniform(-1, 1, size=traj['rewards'].shape)
            assert observations.shape[0] == actions.shape[0] == rewards.shape[0], 'observations, actions, rewards should have the same length'
            self.traj_length[i] = observations.shape[0]

            self.states[self.traj_sp[i]: self.traj_sp[i] + self.traj_length[i]] = observations
            self.actions[self.traj_sp[i]: self.traj_sp[i] + self.traj_length[i]] = actions
            self.rewards_to_go[self.traj_sp[i]: self.traj_sp[i] + self.traj_length[i]] = discounted_cumsum(rewards,  gamma=1.0)
            self.traj_returns[i] = np.sum(rewards)
            traj_pointer += self.traj_length[i]

        assert config.sample_type in ['uniform', 'traj_return', 'traj_length'], 'sample_type should be one of [uniform, traj_return, traj_length]'
        self.p_sample = np.ones(self.num_trajs) / self.num_trajs if  config.sample_type == 'uniform' else self.traj_returns / \
            self.traj_returns.sum() if  config.sample_type == 'traj_return' else self.traj_length / self.traj_length.sum()
        # self.state_mean, self.state_std = self.states.mean(axis=0), self.states.std(axis=0) + 1e-6
        self.state_mean, self.state_std = info["obs_mean"], info["obs_std"]
        self.drop_fn = get_drop_fn(config, self.size, self.traj_sp, self.rng)
            
    def sample(self, batch_size):
        selected_traj = self.rng.choice(np.arange(self.num_trajs), batch_size, replace=True, p=self.p_sample)
        selected_traj_sp = self.traj_sp[selected_traj]
        selected_offset = np.floor(self.rng.random(batch_size) * (self.traj_length[selected_traj] - self.context_len)).astype(np.int32).clip(min=0)
        selected_sp = selected_traj_sp + selected_offset
        selected_ep = selected_sp + self.traj_length[selected_traj].clip(max=self.context_len)

        # fill the index of those padded steps with -1, so that we can fetch the last step of the corresponding item, which is zero intentionally
        selected_index = selected_sp[:, None] + np.arange(self.context_len)
        selected_index = np.where(selected_index < selected_ep[:, None], selected_index, -1)
        masks = selected_index >= 0
        timesteps = selected_offset[:, None] + np.arange(self.context_len)  # we don't care about the timestep for those padded steps
        
        # update and get drop mask
        self.drop_fn.step()
        dropsteps = self.drop_fn.get_dropsteps(selected_index)
        observation_index = selected_index - dropsteps

        states = torch.as_tensor(self.states[observation_index, :]).to(dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.actions[selected_index, :]).to(dtype=torch.float32, device=self.device)
        rewards_to_go = torch.as_tensor(self.rewards_to_go[observation_index, None]).to(dtype=torch.float32, device=self.device)
        timesteps = torch.as_tensor(timesteps).to(dtype=torch.int32, device=self.device)
        dropsteps = torch.as_tensor(dropsteps).to(dtype=torch.int32, device=self.device)

        return states, actions, rewards_to_go, timesteps, dropsteps, masks


class MaskedCausalAttention(nn.Module):
    def __init__(self, hidden_dim, context_len, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.context_len = context_len

        self.q_net = nn.Linear(hidden_dim, hidden_dim)
        self.k_net = nn.Linear(hidden_dim, hidden_dim)
        self.v_net = nn.Linear(hidden_dim, hidden_dim)

        self.proj_net = nn.Linear(hidden_dim, hidden_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((context_len, context_len))
        mask = torch.tril(ones).view(1, 1, context_len, context_len)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))

        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, hidden_dim, context_len, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(hidden_dim, context_len, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, n_heads, n_blocks, hidden_dim, context_len, drop_p, 
                 action_space, state_mean, state_std, reward_scale, max_timestep, drop_aware, device):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # transformer blocks
        self.context_len = context_len
        # blocks = [Block(hidden_dim, 3 * context_len, n_heads, drop_p) for _ in range(n_blocks)]
        # self.blocks = nn.Sequential(*blocks)

        # projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        self.embed_timestep = nn.Embedding(max_timestep, hidden_dim)
        self.embed_dropstep = nn.Embedding(max_timestep, hidden_dim)
        self.embed_rtg = torch.nn.Linear(1, hidden_dim)
        self.embed_state = torch.nn.Linear(state_dim, hidden_dim)

        self.embed_action = torch.nn.Linear(action_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * context_len,
                    embedding_dim=hidden_dim,
                    num_heads=n_heads,
                    attention_dropout=drop_p,
                    residual_dropout=drop_p,
                )
                for _ in range(n_blocks)
            ]
        )

        # prediction heads
        self.predict_rtg = torch.nn.Linear(hidden_dim, 1)
        self.predict_state = torch.nn.Linear(hidden_dim, state_dim)
        self.predict_action = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())

        self.action_space = action_space
        self.action_space.low = torch.as_tensor(self.action_space.low, dtype=torch.float32)
        self.action_space.high = torch.as_tensor(self.action_space.high, dtype=torch.float32)
        self.state_mean = torch.as_tensor(state_mean, dtype=torch.float32)
        self.state_std = torch.as_tensor(state_std, dtype=torch.float32)
        self.reward_scale = reward_scale

        self.max_timestep = max_timestep
        self.drop_aware = drop_aware
        self.to(device)

    def _norm_action(self, action):
        return (action + 1) * (self.action_space.high - self.action_space.low) / 2 + self.action_space.low
    
    def _norm_state(self, state):
        return (state - self.state_mean) / self.state_std
    
    def _norm_reward_to_go(self, reward_to_go):
        return reward_to_go * self.reward_scale

    def to(self, device):
        self.action_space.low = self.action_space.low.to(device)
        self.action_space.high = self.action_space.high.to(device)
        self.state_mean = self.state_mean.to(device)
        self.state_std = self.state_std.to(device)
        return super().to(device)

    def freeze_trunk(self):
        freezed_models = [self.embed_state, self.embed_action, self.embed_rtg, self.embed_timestep, self.blocks, self.embed_ln]
        for model in freezed_models:
            for p in model.parameters():
                p.requires_grad = False

    def forward(self, states, actions, rewards_to_go, timesteps, dropsteps, padding_mask=None):
        states = self._norm_state(states)
        rewards_to_go = self._norm_reward_to_go(rewards_to_go)
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(rewards_to_go) + time_embeddings

        if self.drop_aware:
            drop_embeddings = self.embed_dropstep(dropsteps)
            state_embeddings += drop_embeddings
            returns_embeddings += drop_embeddings
        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(B, 3 * T, self.hidden_dim)

        h = self.embed_ln(h)

        batch_size, seq_len = states.shape[0], states.shape[1]
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        for block in self.blocks:
            h = block(h, padding_mask=padding_mask)
        # transformer and prediction
        # h = self.blocks(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.
        h = h.reshape(B, T, 3, self.hidden_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_rtg(h[:, 2])     # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:, 2])    # predict next state given r, s, a
        action_preds = self.predict_action(h[:, 1])  # predict action given r, s

        # action_preds = self._norm_action(action_preds)
        return state_preds, action_preds, return_preds
    
    def save(self, save_name):
        os.makedirs('models', exist_ok=True)
        torch.save(self.state_dict(), os.path.join('models', f'{save_name}.pt'))
    
    def load(self, load_name):
        self.load_state_dict(torch.load(os.path.join('models', f'{load_name}.pt')))


@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
    env: gym.Env,
    target_return: float,
    attacker: Evaluation_Attacker = None,
    device: str = "cpu",
) -> Tuple[float, float]:
    # parallel evaluation with vectorized environment
    model.eval()

    episodes = 1
    reward, returns = np.zeros(episodes), np.zeros(episodes)
    done_flags = np.zeros(episodes, dtype=np.bool8)

    state_dim = model.state_dim
    act_dim = model.action_dim
    max_timestep = model.max_timestep
    context_len = model.context_len
    timesteps = torch.arange(max_timestep, device=device)
    dropsteps = torch.zeros(max_timestep, device=device, dtype=torch.long)
    state = env.reset()
    if attacker is not None:
        state = attacker.attack_obs(state)
    
    states = torch.zeros((episodes, max_timestep, state_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((episodes, max_timestep, act_dim), dtype=torch.float32, device=device)
    rewards_to_go = torch.zeros((episodes, max_timestep, 1), dtype=torch.float32, device=device)

    reward_to_go, timestep, dropstep = target_return, 0, 0

    while not done_flags.all():
        states[:, timestep] = torch.from_numpy(state).to(device)
        rewards_to_go[:, timestep] = reward_to_go - torch.from_numpy(returns).to(device).unsqueeze(-1)
        dropsteps[timestep] = dropstep
        obs_index = torch.arange(max(0, timestep-context_len+1), timestep+1)
        _, action_preds, _ = model.forward(states[:, obs_index],
                                        actions[:, obs_index],
                                        rewards_to_go[:, obs_index - dropsteps[obs_index].cpu()], # drop rewards
                                        timesteps[None, obs_index],
                                        dropsteps[None, obs_index])

        action = action_preds[:, -1].detach()
        actions[:, timestep] = action

        state, reward, dones, info = env.step(action[0].cpu().numpy())
        if attacker is not None:
            state = attacker.attack_obs(state)
        dropstep = dropsteps[timestep].item() + 1 if info.get('dropped', False) else 0
        returns += reward * ~done_flags
        done_flags = np.bitwise_or(done_flags, dones)
        timestep += 1

    return returns, timestep


def eval_fn(config, env, model, attacker=None):
    eval_log = {}
    for target_return in config.target_returns:
        eval_returns = []
        for _ in trange(config.n_episodes, desc="Evaluation", leave=False):
            eval_return, eval_len = eval_rollout(
                model=model,
                env=env,
                target_return=target_return,
                attacker=attacker,
                device=config.device,
            )
            # unscale for logging & correct normalized score computation
            eval_returns.append(eval_return)

        eval_returns = np.array(eval_returns)
        normalized_score = env.get_normalized_score(eval_returns) * 100
        eval_log.update({
            f"eval/{target_return}_reward_mean": np.mean(eval_returns),
            f"eval/{target_return}_reward_std": np.std(eval_returns),
            f"eval/{target_return}_normalized_score_mean": np.mean(normalized_score),
            f"eval/{target_return}_normalized_score_std": np.std(normalized_score),
        })
    return eval_log


# @pyrallis.wrap()
def train(config: TrainConfig, logger: Logger):
    # Set seeds
    func.set_seed(config.seed)
    if config.use_wandb:
        func.wandb_init(config)

    env = gym.make(config.env)
    env.seed(config.seed)

    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]
    config.max_action = float(env.action_space.high[0])
    config.action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
    ]

    # data & dataloader setup
    dataset = SequenceBuffer(config, logger)
    logger.info(f"Dataset: {dataset.num_trajs} trajectories")
    logger.info(f"State mean: {dataset.state_mean}, std: {dataset.state_std}")

    # model
    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        n_heads=config.num_heads,
        n_blocks=config.num_layers,
        hidden_dim=config.embedding_dim,
        context_len=config.seq_len,
        drop_p=config.model_drop_p,
        action_space=env.action_space,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
        max_timestep=config.episode_len,
        drop_aware=config.drop_aware,
        device=config.device,
    )

    logger.info(f"Network: \n{str(model)}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # optimizer
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )

    model.eval()
    eval_log = eval_fn(config, env, model)
    logger.record("epoch", 0)
    for k, v in eval_log.items():
        logger.record(k, v)
    logger.dump(0)
    if config.use_wandb:
        wandb.log({"epoch": 0, **eval_log})

    model.train()
    total_updates = 0
    best_reward = -np.inf
    # trainloader_iter = iter(trainloader)
    for epoch in trange(1, config.num_epochs + 1, desc="Training"):
        time_start = time.time()
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            log_dict = {}
            
            states, actions, rewards_to_go, timesteps, dropsteps, mask = dataset.sample(config.batch_size)
            padding_mask = torch.from_numpy(~mask).to(dtype=torch.bool, device=config.device)
            # no need for attention mask for the model as we always pad on the right side, whose attention is ignored by the casual mask anyway
            state_preds, action_preds, return_preds = model.forward(states, actions, rewards_to_go, timesteps, dropsteps, padding_mask)
            action_preds = action_preds[mask]
            loss = F.mse_loss(action_preds, actions[mask].detach(), reduction='mean')

            optim.zero_grad()
            loss.backward()
            if config.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optim.step()
            log_dict.update({"loss_action": loss.item()})

            scheduler.step()
            log_dict.update({"learning_rate": scheduler.get_last_lr()[0]})
            total_updates += 1

            if (total_updates + 1) == config.train_steps:
                if config.finetune_steps > 0 and model.drop_aware:
                    logger.info(f"Finetuning seed {config.seed} for {config.finetune_steps} timesteps")
                    model.freeze_trunk()
                    dataset.drop_fn.drop_p = config.finetune_drop_p
                    dataset.drop_fn.update_dropmask()

        time_end = time.time()
        epoch_time = time_end - time_start

        # validation in the env for the actual online performance
        if epoch % config.eval_every == 0:  #  or epoch == config.num_epochs - 1:
            model.eval()
            eval_log = eval_fn(config, env, model)
            model.train()
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

            now_reward = eval_log[f"eval/{config.target_returns[0]}_reward_mean"]
            if now_reward > best_reward:
                best_reward = now_reward
                torch.save(
                    model.state_dict(),
                    os.path.join(logger.get_dir(), f"policy_best.pth"),
                )
                logger.info(
                    f"Save policy on epoch {epoch} for best reward {best_reward}."
                )

        if epoch % 50 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(logger.get_dir(), f"{epoch}.pt"),
            )
            logger.info(f"Save policy on epoch {epoch}.")


def test(config: TrainConfig, logger: Logger):
    # Set seeds
    func.set_seed(config.seed)

    env = gym.make(config.env)
    env.seed(config.seed)

    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]
    config.max_action = float(env.action_space.high[0])
    config.action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
    ]

    # data & dataloader setup
    dataset = SequenceBuffer(config, logger)
    logger.info(f"Dataset: {dataset.num_trajs} trajectories")
    logger.info(f"State mean: {dataset.state_mean}, std: {dataset.state_std}")

    # model
    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        n_heads=config.num_heads,
        n_blocks=config.num_layers,
        hidden_dim=config.embedding_dim,
        context_len=config.seq_len,
        drop_p=config.model_drop_p,
        action_space=env.action_space,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
        max_timestep=config.episode_len,
        drop_aware=config.drop_aware,
        device=config.device,
    )
    model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, "100.pt")))
    model.eval()
    logger.info(f"Network: \n{str(model)}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    attacker = None
    if config.eval_attack:
        state_std = func.get_state_std(config)
        attacker = Evaluation_Attacker(
            config.env, config.corruption_agent, config.eval_attack_eps,
            config.state_dim, config.action_dim, state_std, config.eval_attack_mode
        )

    eval_log = eval_fn(config, env, model, attacker=attacker)
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