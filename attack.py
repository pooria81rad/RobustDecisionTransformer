from typing import Dict

import os
import copy
import gym
import d4rl
import torch
import torch.nn as nn
import numpy as np
import pytorch_util as ptu

from torch.distributions import kl_divergence
from logger import Logger

DATA_NAMSE = {
    "obs": "observations",
    "act": "actions",
    "rew": "rewards",
}

MODEL_PATH = {
    "IQL": os.path.join(os.path.dirname(__file__), "IQL_model"),
}


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


def get_policy_kl(policy, observation, noised_obs):
    _, policy_mean, policy_log_std, _, *_ = policy.stochastic_policy(observation)
    _, noised_policy_mean, noised_policy_log_std, _, *_ = policy.stochastic_policy(noised_obs)
    action_dist = torch.distributions.Normal(policy_mean, policy_log_std.exp())
    noised_action_dist = torch.distributions.Normal(noised_policy_mean, noised_policy_log_std.exp())

    kl_loss = kl_divergence(action_dist, noised_action_dist).sum(axis=-1) + kl_divergence(noised_action_dist, action_dist).sum(axis=-1)
    return kl_loss


def get_policy_mse(policy, observation, noised_obs):
    policy_mean = policy.batch_act(observation)
    noised_policy_mean = policy.batch_act(noised_obs)
    mse_loss = ((policy_mean - noised_policy_mean) ** 2).sum(axis=-1)
    return mse_loss


def optimize_para(para, observation, loss_fun, update_times, step_size, eps, std):
    for i in range(update_times):
        para = torch.nn.Parameter(para.clone(), requires_grad=True)
        optimizer = torch.optim.Adam([para], lr=step_size * eps)
        loss = loss_fun(observation, para)
        # optimize noised obs
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        # para = torch.clamp(para, -eps, eps).detach()
        para = torch.maximum(torch.minimum(para, eps * std), -eps * std).detach()
    return para


class Evaluation_Attacker:
    def __init__(self, env_name, agent_name, eps, obs_dim, action_dim, obs_std=None, attack_mode='random', num_samples=50):
        self.env_name = env_name
        self.agent_name = agent_name
        self.eps = eps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.attack_mode = attack_mode
        self.num_samples = num_samples
        if obs_std is None or isinstance(obs_std, float):
            self.obs_std = torch.ones(1, self.obs_dim, device=ptu.device)
        else:
            self.obs_std = ptu.from_numpy(obs_std)
        # self.obs_std = ptu.from_numpy(obs_std) if obs_std is not None else torch.ones(1, self.obs_dim, device=ptu.device)
        if attack_mode != "random":
            self.load_model()

    def sample_random(self, size):
        return 2 * self.eps * self.obs_std * (torch.rand(size, self.obs_dim, device=ptu.device) - 0.5)

    def noise_action_diff(self, observation, M):
        observation = observation.reshape(M, self.obs_dim)
        size = self.num_samples # for zero order

        def _loss_action(observation, para):
            noised_obs = observation + para
            return - get_policy_mse(self.policy, observation, noised_obs)

        delta_s = self.sample_random(size).reshape(1, size, self.obs_dim).repeat(M, 1, 1).reshape(-1, self.obs_dim)
        tmp_obs = observation.reshape(-1, 1, self.obs_dim).repeat(1, size, 1).reshape(-1, self.obs_dim)
        with torch.no_grad():
            kl_loss = _loss_action(tmp_obs, delta_s)
            max_id = torch.argmin(kl_loss.reshape(M, size), axis=1)
        noise_obs_final = ptu.get_numpy(delta_s.reshape(M, size, self.obs_dim)[np.arange(M), max_id])

        return ptu.get_numpy(observation) + noise_obs_final

    def attack_obs(self, observation):
        M = observation.shape[0] if len(observation.shape) == 2 else 1
        observation = ptu.from_numpy(observation)
        if self.attack_mode == 'random':
            delta_s = self.sample_random(M)
            noised_observation = observation.reshape(M, self.obs_dim) + delta_s
            noised_observation = ptu.get_numpy(noised_observation)
        elif 'action_diff' in self.attack_mode:
            noised_observation = self.noise_action_diff(observation, M)
        else:
            raise NotImplementedError
        return noised_observation.squeeze(0)

    def load_model(self):
        model_path = os.path.join(MODEL_PATH[self.agent_name], self.env_name)
        for root, dirs, files in os.walk(model_path):
            if f"{self.agent_name}_{self.env_name}" in root:
                model_path = os.path.join(root, "3000.pt")
                break
        state_dict = torch.load(model_path, map_location=ptu.device)
        if self.agent_name == "IQL":
            from IQL import DeterministicPolicy, TwinQ

            actor_dropout = None
            key = self.env_name.split("-")[0]
            if key in ["door", "pen", "hammer", "relocate", "kitchen"]:
                actor_dropout = 0.1

            self.policy = (
                DeterministicPolicy(
                    self.obs_dim, self.action_dim, 1.0, n_hidden=2, dropout=actor_dropout
                )
                .to(ptu.device)
                .eval()
            )
            print(str(self.policy))
            self.policy.load_state_dict(state_dict["actor"])
        else:
            raise NotImplementedError


class Attack:
    def __init__(
        self,
        env_name: str,
        agent_name: str,
        dataset: Dict[str, np.ndarray],
        model_path: str,
        dataset_path: str,
        update_times: int = 100,
        step_size: float = 0.01,
        same_index: bool = False,
        froce_attack: bool = False,
        seed: int = 2023,
        device: str = "cpu",
        logger: Logger = None,
    ):
        self.env_name = env_name
        self.agent_name = agent_name
        self.dataset = copy.deepcopy(dataset)
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.update_times = update_times
        self.step_size = step_size
        self.same_index = same_index
        self.froce_attack = froce_attack
        self.seed = seed
        self.device = device
        self.logger = logger

        self._np_rng = np.random.RandomState(seed)
        self._th_rng = torch.Generator()
        self._th_rng.manual_seed(seed)

        self.attack_indexs = None
        self.original_indexs = None

        env = gym.make(env_name)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        env.close()

    def set_attack_config(
        self,
        corruption_name,
        corruption_tag,
        corruption_rate,
        corruption_range,
        corruption_random,
    ):
        self.corruption_tag = DATA_NAMSE[corruption_tag]
        self.corruption_rate = corruption_rate
        self.corruption_range = corruption_range
        self.corruption_random = corruption_random
        self.new_dataset_path = os.path.expanduser(
            os.path.join(self.dataset_path, "log_attack_data", self.env_name)
        )
        self.new_dataset_file = (
            f"random_{self.seed}{corruption_name}.pth"
            if self.corruption_random
            else f"{self.agent_name}_adversarial{corruption_name}.pth"
        )

        self.corrupt_func = getattr(self, f"corrupt_{corruption_tag}")
        self.loss_Q = getattr(self, f"loss_Q_for_{corruption_tag}")
        if self.attack_indexs is None or not self.same_index:
            self.attack_indexs, self.original_indexs = self.sample_indexs()

    def load_model(self):
        model_path = os.path.join(self.model_path, self.env_name)
        for root, dirs, files in os.walk(model_path):
            if f"{self.agent_name}_{self.env_name}" in root:
                model_path = os.path.join(root, "3000.pt")
                break
        state_dict = torch.load(model_path, map_location=self.device)
        if self.agent_name == "IQL":
            from IQL import DeterministicPolicy, TwinQ

            self.critic = (
                TwinQ(self.state_dim, self.action_dim, n_hidden=2)
                .to(self.device)
                .eval()
            )
            # self.actor.load_state_dict(state_dict["actor"])
            self.critic.load_state_dict(state_dict["qf"])
        else:
            raise NotImplementedError
        self.logger.info(f"Load model from {model_path}")

    def optimize_para(self, para, std, obs, act=None):
        for _ in range(self.update_times):
            para = torch.nn.Parameter(para.clone(), requires_grad=True)
            optimizer = torch.optim.Adam(
                [para], lr=self.step_size * self.corruption_range
            )
            loss = self.loss_Q(para, obs, act, std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            para = torch.clamp(
                para, -self.corruption_range, self.corruption_range
            ).detach()
        return para * std

    def loss_Q_for_obs(self, para, observation, action, std):
        noised_obs = observation + para * std
        qvalue = self.critic(noised_obs, action)
        return qvalue.mean()

    def loss_Q_for_act(self, para, observation, action, std):
        noised_act = action + para * std
        qvalue = self.critic(observation, noised_act)
        return qvalue.mean()

    def loss_Q_for_next_obs(self, para, observation, action, std):
        noised_obs = observation + para * std
        action = self.actor.batch_act(noised_obs, self.device)
        qvalue = self.critic(noised_obs, action)
        return qvalue.mean()

    def loss_Q_for_rew(self):
        # Just Placeholder
        raise NotImplementedError

    def sample_indexs(self):
        indexs = np.arange(len(self.dataset["rewards"]))
        random_num = self._np_rng.random(len(indexs))
        attacked = np.where(random_num < self.corruption_rate)[0]
        original = np.where(random_num >= self.corruption_rate)[0]
        return indexs[attacked], indexs[original]

    def sample_para(self, data, std):
        return (
            2
            * self.corruption_range
            * std
            * (torch.rand(data.shape, generator=self._th_rng).to(self.device) - 0.5)
        )

    def sample_data(self, data):
        random_data = self._np_rng.uniform(
            -self.corruption_range, self.corruption_range, size=data.shape
        )
        return random_data

    def corrupt_obs(self, dataset):
        # load original obs
        original_obs = self.dataset[self.corruption_tag][self.attack_indexs].copy()
        std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)

        if self.corruption_random:
            attack_obs = original_obs + self.sample_data(original_obs) * std
            self.logger.info(f"Random attack {self.corruption_tag}")
        else:
            self.load_model()

            std_torch = torch.from_numpy(std).to(self.device)
            original_act = self.dataset["actions"][self.attack_indexs].copy()
            original_act_torch = torch.from_numpy(original_act.copy()).to(self.device)
            original_obs_torch = torch.from_numpy(original_obs.copy()).to(self.device)

            # adversarial attack obs
            attack_obs = np.zeros_like(original_obs)
            split = 10
            pointer = 0
            M = original_obs.shape[0]
            for i in range(split):
                number = M // split if i < split - 1 else M - pointer
                temp_act = original_act_torch[pointer : pointer + number]
                temp_obs = original_obs_torch[pointer : pointer + number]
                para = self.sample_para(temp_obs, std_torch)
                para = self.optimize_para(para, std_torch, temp_obs, temp_act)
                noise = para.cpu().numpy()
                attack_obs[pointer : pointer + number] = noise + temp_obs.cpu().numpy()
                pointer += number

            self.clear_gpu_cache()
            self.logger.info(f"Adversarial attack {self.corruption_tag}")

        self.save_dataset(attack_obs)
        dataset[self.corruption_tag][self.attack_indexs] = attack_obs
        return dataset

    def corrupt_act(self, dataset):
        # load original act
        original_act = self.dataset[self.corruption_tag][self.attack_indexs].copy()
        std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)

        if self.corruption_random:
            attack_act = original_act + self.sample_data(original_act) * std
            self.logger.info(f"Random attack {self.corruption_tag}")
        else:
            self.load_model()

            std_torch = torch.from_numpy(std).to(self.device)
            original_obs = self.dataset["observations"][self.attack_indexs].copy()
            original_obs_torch = torch.from_numpy(original_obs.copy()).to(self.device)
            original_act_torch = torch.from_numpy(original_act.copy()).to(self.device)

            # adversarial attack act
            attack_act = np.zeros_like(original_act)
            split = 10
            pointer = 0
            M = original_act.shape[0]
            for i in range(split):
                number = M // split if i < split - 1 else M - pointer
                temp_obs = original_obs_torch[pointer : pointer + number]
                temp_act = original_act_torch[pointer : pointer + number]
                para = self.sample_para(temp_act, std_torch)
                para = self.optimize_para(para, std_torch, temp_obs, temp_act)
                noise = para.cpu().numpy()
                attack_act[pointer : pointer + number] = noise + temp_act.cpu().numpy()
                pointer += number

            self.clear_gpu_cache()
            self.logger.info(f"Adversarial attack {self.corruption_tag}")
        self.save_dataset(attack_act)
        dataset[self.corruption_tag][self.attack_indexs] = attack_act
        return dataset

    def corrupt_rew(self, dataset):
        # load original rew
        original_rew = self.dataset[self.corruption_tag][self.attack_indexs].copy()
        std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)

        if self.corruption_random:
            attack_rew = self._np_rng.uniform(
                -self.corruption_range, self.corruption_range, size=original_rew.shape
            )
            self.logger.info(f"Random attack {self.corruption_tag}")
        else:
            attack_rew = original_rew.copy() * -self.corruption_range
            self.logger.info(f"Adversarial attack {self.corruption_tag}")

        self.save_dataset(attack_rew)
        dataset[self.corruption_tag][self.attack_indexs] = attack_rew
        return dataset

    def corrupt_next_obs(self, dataset):
        # load original obs
        original_obs = self.dataset[self.corruption_tag][self.attack_indexs].copy()
        std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)

        if self.corruption_random:
            std = np.std(self.dataset[self.corruption_tag], axis=0, keepdims=True)
            attack_obs = original_obs + self.sample_data(original_obs) * std
            self.logger.info(f"Random attack {self.corruption_tag}")
        else:
            self.load_model()

            std_torch = torch.from_numpy(std).to(self.device)
            original_obs_torch = torch.from_numpy(original_obs.copy()).to(self.device)

            # adversarial attack obs
            attack_obs = np.zeros_like(original_obs)
            split = 10
            pointer = 0
            M = original_obs.shape[0]
            for i in range(split):
                number = M // split if i < split - 1 else M - pointer
                temp_obs = original_obs_torch[pointer : pointer + number]
                para = self.sample_para(temp_obs, std_torch)
                para = self.optimize_para(para, std_torch, temp_obs)
                noise = para.cpu().numpy()
                attack_obs[pointer : pointer + number] = noise + temp_obs.cpu().numpy()
                pointer += number

            self.clear_gpu_cache()
            self.logger.info(f"Adversarial attack {self.corruption_tag}")

        self.save_dataset(attack_obs)
        dataset[self.corruption_tag][self.attack_indexs] = attack_obs
        return dataset

    def clear_gpu_cache(self):
        # self.actor.to("cpu")
        self.critic.to("cpu")
        torch.cuda.empty_cache()

    def save_dataset(self, attack_datas):
        ### save data
        save_dict = {}
        save_dict["attack_indexs"] = self.attack_indexs
        save_dict["original_indexs"] = self.original_indexs
        save_dict[self.corruption_tag] = attack_datas
        if not os.path.exists(self.new_dataset_path):
            os.makedirs(self.new_dataset_path)
        dataset_path = os.path.join(self.new_dataset_path, self.new_dataset_file)
        torch.save(save_dict, dataset_path)
        self.logger.info(f"Save attack dataset in {dataset_path}")

    def get_original_data(self, indexs):
        dataset = {}
        dataset["observations"] = self.dataset["observations"][indexs]
        dataset["actions"] = self.dataset["actions"][indexs]
        dataset["rewards"] = self.dataset["rewards"][indexs]
        if "next_observations" in self.dataset.keys():
            dataset["next_observations"] = self.dataset["next_observations"][indexs]
        dataset["terminals"] = self.dataset["terminals"][indexs]
        return dataset

    def attack(self, dataset):
        dataset_path = os.path.join(self.new_dataset_path, self.new_dataset_file)
        if os.path.exists(dataset_path) and not self.froce_attack:
            new_dataset = torch.load(dataset_path)
            self.logger.info(f"Load new dataset from {dataset_path}")
            original_indexs, attack_indexs, attack_datas = (
                new_dataset["original_indexs"],
                new_dataset["attack_indexs"],
                new_dataset[self.corruption_tag],
            )
            ori_dataset = self.get_original_data(original_indexs)
            dataset[self.corruption_tag][attack_indexs] = attack_datas
            self.attack_indexs = attack_indexs
            return ori_dataset, dataset
        else:
            ori_dataset = self.get_original_data(self.original_indexs)
            att_dataset = self.corrupt_func(dataset)
            return ori_dataset, att_dataset


def attack_dataset(config, dataset, logger):
    attack_agent = Attack(
        env_name=config.env,
        agent_name=config.corruption_agent,
        dataset=dataset,
        model_path=MODEL_PATH[config.corruption_agent],
        dataset_path=config.dataset_path,
        same_index=config.same_index,
        froce_attack=config.froce_attack,
        seed=config.corruption_seed,
        device=config.device,
        logger=logger,
    )
    corruption_random = config.corruption_mode == "random"
    attack_params = {
        "corruption_rate": config.corruption_rate,
        "corruption_random": corruption_random,
    }
    name = ""
    if config.sample_ratio < 1:
        name += f"_ratio_{config.sample_ratio}"
    attack_indexs = []
    if config.corruption_obs > 0:
        name += f"_obs_{config.corruption_obs}_{config.corruption_rate}"
        attack_params["corruption_range"] = config.corruption_obs
        attack_agent.set_attack_config(name, "obs", **attack_params)
        ori_dataset, att_dataset = attack_agent.attack(dataset)
        dataset = ori_dataset if config.use_original else att_dataset
        attack_indexs.append(attack_agent.attack_indexs)
        logger.info(f"{config.corruption_mode} observations")
    if config.corruption_act > 0:
        name += f"_act_{config.corruption_act}_{config.corruption_rate}"
        attack_params["corruption_range"] = config.corruption_act
        attack_agent.set_attack_config(name, "act", **attack_params)
        ori_dataset, att_dataset = attack_agent.attack(dataset)
        dataset = ori_dataset if config.use_original else att_dataset
        attack_indexs.append(attack_agent.attack_indexs)
        logger.info(f"{config.corruption_mode} actions")
    if config.corruption_rew > 0:
        name += f"_rew_{config.corruption_rew}_{config.corruption_rate}"
        attack_params["corruption_range"] = config.corruption_rew
        attack_agent.set_attack_config(name, "rew", **attack_params)
        ori_dataset, att_dataset = attack_agent.attack(dataset)
        dataset = ori_dataset if config.use_original else att_dataset
        attack_indexs.append(attack_agent.attack_indexs)
        logger.info(f"{config.corruption_mode} rewards")
    logger.info(f"Attack name: {name}")

    attack_indexs = np.hstack(attack_indexs)
    return dataset, attack_indexs
