import os
import gym
import d4rl
import torch
import random
import numpy as np
from tqdm import trange
from collections import defaultdict

def ratio_dataset(dataset_path, env_name, ratio):
    random.seed(1234)
    np.random.seed(1234)

    h5path = os.path.expanduser(f"{dataset_path}/{env_name}.hdf5")
    dataset = gym.make(env_name).get_dataset(h5path=h5path)

    traj, traj_len = [], []
    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])
        data_["terminals"].append(dataset["terminals"][i])
        data_["timeouts"].append(dataset["timeouts"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            traj.append(data_)
            traj_len.append(len(data_["actions"]))
            # reset trajectory buffer
            data_ = defaultdict(list)

    new_traj = random.sample(traj, int(len(traj)*ratio))

    new_dataset = {
        "observations": np.concatenate([traj["observations"] for traj in new_traj], axis=0),
        "actions": np.concatenate([traj["actions"] for traj in new_traj], axis=0),
        "rewards": np.concatenate([traj["rewards"] for traj in new_traj], axis=0),
        "terminals": np.concatenate([traj["terminals"] for traj in new_traj], axis=0),
        "timeouts": np.concatenate([traj["timeouts"] for traj in new_traj], axis=0),
    }
    # os.mkdir(os.path.join(dataset_path, "original"))
    dataset_path = os.path.join(dataset_path, "original", f"{env_name}_ratio_{ratio}.pt")
    torch.save(new_dataset, dataset_path)
    print(f"Save downsample dataset in {dataset_path}")
    print(f"=============================================")
    print(f"Env: {env_name}")
    print(f"Trajectory number: {len(traj)} -> {len(new_traj)}")
    data_num = np.sum([len(traj["actions"]) for traj in traj])
    new_data_num = np.sum([len(traj["actions"]) for traj in new_traj])
    print(f"Transition number: {data_num} -> {new_data_num}")
    print(f"=============================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="walker2d-medium-replay-v2")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--dataset_path", type=str, default="your_path_of_dataset")
    args = parser.parse_args()
    ratio_dataset(args.dataset_path, args.env_name, args.ratio)
