from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import os
import gym
import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict
from tqdm.auto import trange  # noqa

from networks import MLPBlock, ResidualBlock, TransformerBlock
from attack import attack_dataset, Evaluation_Attacker
from logger import Logger

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
    config, env_name: str, gamma: float = 1.0, logger: Logger = None
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
    # return ori_traj, traj, info
    return traj, info


class SequenceDataset:
    def __init__(self, config, logger: Logger = None):
        self.dataset, info = load_d4rl_trajectories(config, config.env, gamma=1.0, logger=logger)
        self.seq_len = config.seq_len
        self.max_ep_len = config.episode_len
        self.reward_scale = config.reward_scale
        self.recalculate_return = config.recalculate_return
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.action_range = config.action_range

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()
        self.float_dtype = np.float32

    def correct(self, traj_indexs, time_steps, correct_data, correct_type):
        for i, (tarj_i, step_j) in enumerate(zip(traj_indexs, time_steps)):
            if step_j < self.dataset[tarj_i][correct_type].shape[0]:
                self.dataset[tarj_i][correct_type][step_j] = correct_data[i]

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        if self.recalculate_return:
            returns = discounted_cumsum(traj["rewards"][start_idx :], gamma=1.0).reshape(-1, 1)[: self.seq_len]
        else:
            returns = traj["returns"][start_idx : start_idx + self.seq_len].reshape(-1, 1)
        rewards = traj["rewards"][start_idx : start_idx + self.seq_len].reshape(-1, 1)
        attack_mask = traj["attack_mask"][start_idx : start_idx + self.seq_len].reshape(-1, 1)
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        rewards = rewards * self.reward_scale
        # pad up to seq_len if needed
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)
            rewards = pad_along_axis(rewards, pad_to=self.seq_len)
            attack_mask = pad_along_axis(attack_mask, pad_to=self.seq_len)

        return states.astype(self.float_dtype), actions.astype(self.float_dtype), returns.astype(self.float_dtype), \
            rewards.astype(self.float_dtype), time_steps, mask, attack_mask, traj_idx

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        traj_ids = np.random.choice(np.arange(len(self.dataset)), size=batch_size, p=self.sample_prob, replace=True)

        states, actions, returns, rewards, time_steps, masks, attack_mask, traj_index = [], [], [], [], [], [], [], []
        for traj_id in traj_ids:
            start_idx = np.random.randint(0, self.dataset[traj_id]["rewards"].shape[0])
            state, action, ret, reward, time_step, mask, att_mask, traj_i = self.__prepare_sample(traj_id, start_idx)
            states.append(state)
            actions.append(action)
            returns.append(ret)
            rewards.append(reward)
            time_steps.append(time_step)
            masks.append(mask)
            attack_mask.append(att_mask)
            traj_index.append(traj_i)
        return [torch.tensor(states), torch.tensor(actions), torch.tensor(returns), torch.tensor(rewards), torch.tensor(time_steps), torch.tensor(masks), torch.tensor(attack_mask), torch.tensor(traj_index)]

    def save(self, path: str):
        torch.save(self.dataset, path)


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: nn.Module,
    env: gym.Env,
    target_return: float,
    float_dtype: torch.dtype = torch.float32,
    attacker: Evaluation_Attacker = None,
    device: str = "cpu",
) -> Tuple[float, float]:
    action_range = [
        float(env.action_space.low.min()) + 1e-6,
        float(env.action_space.high.max()) - 1e-6,
    ]
    states = torch.zeros(
        1, model.episode_len + 1, model.state_dim, dtype=float_dtype, device=device
    )
    actions = torch.zeros(
        1, model.episode_len, model.action_dim, dtype=float_dtype, device=device
    )
    returns = torch.zeros(1, model.episode_len + 1, 1, dtype=float_dtype, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    obs = env.reset()
    if attacker is not None:
        obs = attacker.attack_obs(obs)
    states[:, 0] = torch.as_tensor(obs, device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)
        predicted = model(  # fix this noqa!!!
            states[:, : step + 1][:, -model.seq_len :],
            actions[:, : step + 1][:, -model.seq_len :],
            returns[:, : step + 1][:, -model.seq_len :],
            time_steps[:, : step + 1][:, -model.seq_len :],
        )
        predicted_actions = predicted[0]
        predicted_action = predicted_actions[0, -1].cpu().numpy()
        predicted_action = np.clip(predicted_action, *action_range)
        next_state, reward, done, info = env.step(predicted_action)
        if attacker is not None:
            next_state = attacker.attack_obs(next_state)
        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

        episode_return += reward
        episode_len += 1

        if done:
            break
    return episode_return, episode_len


def eval_fn(config, env, model, attacker=None):
    eval_log = {}
    for target_return in config.target_returns:
        eval_returns = []
        for _ in trange(config.n_episodes, desc="Evaluation", leave=False):
            eval_return, eval_len = eval_rollout(
                model=model,
                env=env,
                target_return=target_return * config.reward_scale,
                attacker=attacker,
                device=config.device,
            )
            # unscale for logging & correct normalized score computation
            eval_returns.append(eval_return / config.reward_scale)

        eval_returns = np.array(eval_returns)
        normalized_score = env.get_normalized_score(eval_returns) * 100
        eval_log.update({
            f"eval/{target_return}_reward_mean": np.mean(eval_returns),
            f"eval/{target_return}_reward_std": np.std(eval_returns),
            f"eval/{target_return}_normalized_score_mean": np.mean(normalized_score),
            f"eval/{target_return}_normalized_score_std": np.std(normalized_score),
        })
    return eval_log


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        predict_dropout: float = 0.1,
        mlp_embedding: bool = False,
        mlp_head: bool = False,
        mlp_reward: bool = False,
        predict_reward: bool = False,
        embed_order: str = "rsa",
    ):
        super().__init__()

        if embedding_dropout is not None:
            self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        if mlp_embedding:
            self.state_emb = ResidualBlock(state_dim, embedding_dim)
            self.action_emb = ResidualBlock(action_dim, embedding_dim)
            self.return_emb = ResidualBlock(1, embedding_dim)
        else:
            self.state_emb = nn.Linear(state_dim, embedding_dim)
            self.action_emb = nn.Linear(action_dim, embedding_dim)
            self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.predict_dropout = nn.Dropout(predict_dropout)

        num_layer = 2 if mlp_head else 1
        self.action_head = MLPBlock(embedding_dim, action_dim, num_layer, True)
        if predict_reward:
            num_layer = 2 if mlp_reward else 1
            self.reward_head = MLPBlock(embedding_dim, 1, num_layer)

        self.apply(self._init_weights)

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.embed_order = embed_order
        self.predict_reward = predict_reward

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        if hasattr(self, "state_dropout"):
            states = self.state_dropout(states)
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states)
        act_emb = self.action_emb(actions)
        returns_emb = self.return_emb(returns_to_go)

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        if self.embed_order == "rsa":
            sequence = torch.stack([returns_emb, state_emb, act_emb], dim=1)
        elif self.embed_order == "sar":
            sequence = torch.stack([state_emb, act_emb, returns_emb], dim=1)
        else:
            raise ValueError(f"Invalid embedding order {self.embed_order}.")
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.embedding_dim)
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
        if hasattr(self, "emb_drop"):
            out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        out = self.predict_dropout(out)

        if self.embed_order == "rsa":
            out_r_emb, out_s_emb, out_a_emb = out[:, 0::3], out[:, 1::3], out[:, 2::3]
        elif self.embed_order == "sar":
            out_s_emb, out_a_emb, out_r_emb = out[:, 0::3], out[:, 1::3], out[:, 2::3]
        else:
            raise ValueError(f"Invalid embedding order {self.embed_order}.")

        action_out = self.action_head(out_s_emb)
        if self.predict_reward:
            reward_out = self.reward_head(out_a_emb)
        else:
            reward_out = None
        return action_out, reward_out
