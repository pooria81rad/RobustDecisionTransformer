# inspiration:
# 1. https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py  # noqa
# 2. https://github.com/karpathy/minGPT
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import Optional, Tuple

import traceback
import time
import json
import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import torch
import wandb
import utils.functions as func
import utils.dt_functions as dt_func

from torch.nn import functional as F
from tqdm.auto import trange  # noqa
from dataclasses import dataclass
from utils.logger import init_logger, Logger
from utils.attack import Evaluation_Attacker


@dataclass
class TrainConfig:
    # Experiment
    eval_every: int = 1 # Log every n epochs
    n_episodes: int = 10  # How many episodes run during evaluation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 100
    num_updates_on_epoch: int = 1000
    # model params
    embedding_dim: int = 128 # 768
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.0
    residual_dropout: float = 0.1
    embedding_dropout: float = None
    # training params
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    update_steps: int = 100_000
    warmup_steps: int = 10_000
    reward_scale: float = 0.001
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    recalculate_return: bool = False
    # evaluation params
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    eval_id: str = "00"
    eval_only: bool = False
    eval_attack: bool = True
    eval_attack_eps: float = 0.01
    eval_attack_mode: str = "random"
    checkpoint_dir: str = None
    # Wandb logging
    use_wandb: int = 0
    group: str = "2023082100"
    env: str = "walker2d-medium-replay-v2"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    # dataset
    down_sample: bool = True
    sample_ratio: float = 1.0
    ######## others
    debug: bool = False
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
        if self.down_sample:
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
        # others
        if self.env.startswith("antmaze"):
            self.num_epochs = 100
            self.n_episodes = 100
        if self.corruption_mode == "random" and self.corruption_rew > 0.0:
            self.corruption_rew *= 30
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


def set_model(config: TrainConfig):
    model = dt_func.DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
    ).to(config.device)
    return model


def train(config: TrainConfig, logger: Logger):
    # Set seeds
    func.set_seed(config.seed)

    if config.use_wandb:
        func.wandb_init(config)

    env = gym.make(config.env)
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]
    config.max_action = float(env.action_space.high[0])
    config.action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
    ]

    # data & dataloader setup
    dataset = dt_func.SequenceDataset(config, logger)
    logger.info(f"Dataset: {len(dataset.dataset)} trajectories")
    logger.info(f"State mean: {dataset.state_mean}, std: {dataset.state_std}")

    env = func.wrap_env(
        env,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    env.seed(config.seed)

    # model
    model = set_model(config)
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
    eval_log = dt_func.eval_fn(config, env, model)
    model.train()
    logger.record("epoch", 0)
    for k, v in eval_log.items():
        logger.record(k, v)
    logger.dump(0)
    if config.use_wandb:
        wandb.log({"epoch": 0, **eval_log})

    total_updates = 0
    best_reward = -np.inf
    # trainloader_iter = iter(trainloader)
    for epoch in trange(1, config.num_epochs + 1, desc="Training"):
        time_start = time.time()
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            log_dict = {}
            # batch = next(trainloader_iter)
            batch = dataset.get_batch(config.batch_size)
            states, actions, returns, _, time_steps, mask, _, _ = [b.to(config.device) for b in batch]
            # True value indicates that the corresponding key value will be ignored
            padding_mask = ~mask.to(torch.bool)

            predicted = model(
                states=states,
                actions=actions,
                returns_to_go=returns,
                time_steps=time_steps,
                padding_mask=padding_mask,
            )
            predicted_actions = predicted[0]

            optim.zero_grad()
            loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
            # [batch_size, seq_len, d] * [batch_size, seq_len, 1]
            loss = (loss * mask.unsqueeze(-1)).mean()
            loss.backward()
            if config.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optim.step()
            log_dict.update({"loss_action": loss.item()})

            scheduler.step()
            log_dict.update({"learning_rate": scheduler.get_last_lr()[0]})
            total_updates += 1

        time_end = time.time()
        epoch_time = time_end - time_start

        # validation in the env for the actual online performance
        if epoch % config.eval_every == 0:  #  or epoch == config.num_epochs - 1:
            model.eval()
            eval_log = dt_func.eval_fn(config, env, model)
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
            if config.save_model and now_reward > best_reward:
                best_reward = now_reward
                torch.save(
                    model.state_dict(),
                    os.path.join(logger.get_dir(), f"policy_best.pth"),
                )
                logger.info(
                    f"Save policy on epoch {epoch} for best reward {best_reward}."
                )

        if config.save_model and epoch % 50 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(logger.get_dir(), f"{epoch}.pt"),
            )
            logger.info(f"Save policy on epoch {epoch}.")


def test(config: TrainConfig, logger: Logger):
    # Set seeds
    func.set_seed(config.seed)

    env = gym.make(config.env)
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]
    config.max_action = float(env.action_space.high[0])
    config.action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
    ]

    # data & dataloader setup
    dataset = dt_func.SequenceDataset(config, logger)
    logger.info(f"Dataset: {len(dataset.dataset)} trajectories")
    logger.info(f"State mean: {dataset.state_mean}, std: {dataset.state_std}")

    env = func.wrap_env(
        env,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    env.seed(config.seed)

    # model
    model = set_model(config)
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

    eval_log = dt_func.eval_fn(config, env, model, attacker=attacker)
    for k, v in eval_log.items():
        logger.record(k, v)
    logger.dump(0)


@pyrallis.wrap()
def main(config: TrainConfig):
    logger = init_logger(config)
    # train(config, logger)
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