import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from typing import Any, Dict, Union, Tuple, Optional
import gym
import uuid
import wandb
import random
import numpy as np
import torch
import torch.nn as nn
from attack import Evaluation_Attacker


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.abs(tau - (u < 0).float()) * u**2


def asymmetric_l1_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.abs(tau - (u < 0).float()) * torch.abs(u)


def l2_loss(u: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return u**2


def l1_loss(u: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return torch.abs(u)


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


def set_seed(seed: int, env: Optional[gym.Env] = None):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(deterministic_torch)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int, attacker: Evaluation_Attacker = None
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        if attacker is not None:
            state = attacker.attack_obs(state)
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            if attacker is not None:
                state = attacker.attack_obs(state)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def eval(config, env, actor, attacker=None):
    eval_scores = eval_actor(
        env,
        actor,
        device=config.device,
        n_episodes=config.eval_episodes,
        seed=config.seed,
        attacker=attacker,
    )
    eval_returns = eval_scores
    eval_log = {
        "eval/reward_mean": np.mean(eval_returns),
        "eval/reward_std": np.std(eval_returns),
    }
    normalized_score = env.get_normalized_score(eval_scores) * 100.0
    eval_log["eval/normalized_score_mean"] = np.mean(normalized_score)
    eval_log["eval/normalized_score_std"] = np.std(normalized_score)
    return eval_log


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


def modify_reward_for_CQL(
    dataset: Dict,
    env_name: str,
    max_episode_steps: int = 1000,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def normalize_dataset(config, dataset):
    if config.normalize_reward:
        if config.alg_type == "CQL":
            modify_reward_for_CQL(
                dataset, config.env,
                reward_scale=config.reward_scale, reward_bias=config.reward_bias
                )
        else:
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

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    if "next_observations" in dataset.keys():
        dataset["next_observations"] = normalize_states(
            dataset["next_observations"], state_mean, state_std
        )
    return dataset, state_mean, state_std


def asdict(config):
    dic = {}
    config_dict = config.__dict__
    for key, value in config_dict.items():
        if not key.startswith("__"):
            dic[key] = value
    return dic


def wandb_init(config) -> None:
    corruption_info = config.corruption_mode
    if config.corruption_mode != "none":
        if config.use_original:
            corruption_info += f"_only_original"
            corruption_info += f"_{1 - config.corruption_rate}"
        else:
            if config.corruption_obs > 0:
                corruption_info += f"_obs_{config.corruption_obs}"
            if config.corruption_act > 0:
                corruption_info += f"_act_{config.corruption_act}"
            if config.corruption_rew > 0:
                corruption_info += f"_rew_{config.corruption_rew}"
            if config.corruption_next_obs > 0:
                corruption_info += f"_next_obs_{config.corruption_next_obs}"
            corruption_info += f"_{config.corruption_rate}"
    name = f"{config.alg_type}-{corruption_info}-{config.seed}"
    group = f"{config.group}-{config.env.split('-')[0]}-{corruption_info}"
    project = "CorruptionOffline"
    wandb_dir = os.path.expanduser(os.path.join(config.logdir, config.group))
    wandb.init(
        config=asdict(config),
        project=project,
        group=group,
        name=name,
        id=str(uuid.uuid4()),
        dir=wandb_dir,
        mode="offline",
        save_code=True,
    )
    wandb.run.save()


def load_clean_dataset(config):
    if config.sample_ratio < 1.0:
        dataset_path = os.path.join(config.dataset_path, "original", f"{config.env}_ratio_{config.sample_ratio}.pt")
        dataset = torch.load(dataset_path)
    else:
        h5path = (
            config.dataset_path
            if config.dataset_path is None
            else os.path.expanduser(f"{config.dataset_path}/{config.env}.hdf5")
        )
        dataset = gym.make(config.env).get_dataset(h5path=h5path)
    return dataset


def get_state_std(config):
    clean_dataset = load_clean_dataset(config)
    state_std = compute_mean_std(clean_dataset["observations"], eps=1e-3)[1]
    return state_std
