import logging
import os
import random

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)


class WAIRDDatasetSequence(Dataset):
    
    def __init__(
        self, data_path: str, scenario: str, scenario2_path: str, split: str,
        use_channels: list[int], n_links: int, los_ratio: float
    ):
        super().__init__()
        
        if scenario != "scenario_1":
            raise NotImplementedError(f'There is only an implementation for scenario_1. Given {scenario}')
        
        if split != "train" and split != "val" and split != "test":
            raise NotImplementedError(f'There is only an implementation for train and test. Given {split}')
        
        self.__scenario_path: str = os.path.join(data_path, scenario)
        self.__scenario2_path: str = scenario2_path
        self.__split: str = split
        self.__use_channels = use_channels
        self.__n_links = n_links
        self.__los_ratio = los_ratio
        
        self.__num_envs = 10000 - 1
        
        self.__num_train_envs_count = 10000 - 500 - 100 - 1 - 96
        self.__num_val_envs_count = 500 + 96
        self.__num_test_envs_count = 100
        
        self.__num_bss_per_env = 5
        self.__num_ues_per_env = 30
        self.__num_pairs_per_env = self.__num_bss_per_env * self.__num_ues_per_env
        
        self.__environments: list[str] = self.__prepare_environments()
        self.__chosen_idx = self.__subsample_los()
    
    def __getitem__(self, data_idx: int):
        if self.__split == "train" and self.__los_ratio < 1:
            pair_idx = self.__chosen_idx[data_idx]
        else:
            pair_idx = data_idx
        
        environment_idx: int = pair_idx // self.__num_pairs_per_env
        local_pair_idx = pair_idx % self.__num_pairs_per_env
        bs_idx = local_pair_idx // self.__num_ues_per_env
        ue_idx = local_pair_idx % self.__num_ues_per_env
        environment: str = self.__environments[environment_idx]
        env_path = os.path.join(self.__scenario_path, environment)
        
        pair_path = os.path.join(env_path, f"bs{bs_idx}_ue{ue_idx}")
        pair_data = np.load(os.path.join(pair_path, "pairs_data.npz"), allow_pickle=True)
        
        img_size = pair_data["metadata"].item()["img_size"]
        locations = pair_data["locations"].item()
        ue_location = locations['ue'].astype(np.float32)[:2].astype(np.float32)
        bs_location = locations['bs'].astype(np.float32)[:2].astype(np.float32)
        is_los = pair_data["is_los"]
        
        sequence = pair_data["sequence"].astype(np.float32)
        sequence = np.concatenate([np.tile(bs_location, sequence.shape[0]).reshape(-1, 2), sequence], axis=1)
        
        if self.__n_links is not None:
            pad_len = max(0, self.__n_links - sequence.shape[0])
            mask = np.array([0] * min(sequence.shape[0], self.__n_links) + [1] * pad_len, dtype=np.bool_)
            pad = np.zeros((pad_len, sequence.shape[1]))
            sequence = np.vstack([sequence, pad])
            sequence = sequence[: self.__n_links]
        else:
            mask = np.array([0] * sequence.shape[0])
        
        if self.__use_channels is not None:
            sequence = sequence[:, self.__use_channels]
        sequence = sequence.astype(np.float32)
        
        return sequence, mask, ue_location, img_size / 2, is_los
    
    def __len__(self):
        if self.__split == "train":
            if self.__los_ratio < 1:
                return len(self.__chosen_idx)
            env_count = self.__num_train_envs_count
        elif self.__split == "val":
            env_count = self.__num_val_envs_count
        else:
            env_count = self.__num_test_envs_count
        
        return env_count * self.__num_bss_per_env * self.__num_ues_per_env
    
    def __prepare_environments(self) -> list[str]:
        environments = os.listdir(self.__scenario_path)
        environments = sorted(filter(str.isnumeric, environments))
        scenario2_environments = set(filter(str.isnumeric, os.listdir(self.__scenario2_path)))
        
        if self.__split == "train":
            return sorted(set(environments[:900] + environments[1000: 9499]) - scenario2_environments)
        elif self.__split == "val":
            return sorted(set(environments[9499:]) | scenario2_environments)
        else:
            return environments[900:1000]
    
    def __subsample_los(self) -> list[int]:
        if self.__split == "train" and self.__los_ratio < 1:
            log.info("Subsampling LOS data")
            is_los_list = []
            for env in tqdm(self.__environments):
                env_path = os.path.join(self.__scenario_path, env)
                for bs_idx in range(self.__num_bss_per_env):
                    for ue_idx in range(self.__num_ues_per_env):
                        pair_path = os.path.join(env_path, f"bs{bs_idx}_ue{ue_idx}")
                        pair_data = np.load(os.path.join(pair_path, "pairs_data.npz"), allow_pickle=True)
                        is_los = pair_data["is_los"]
                        is_los_list.append(is_los)
            is_los_list = np.array(is_los_list)
            los_idx = np.where(is_los_list)[0]
            nlos_idx = np.where(~is_los_list)[0]
            # num_los = int(los_idx.shape[0] * los_ratio)
            num_los = int((is_los_list.shape[0] - los_idx.shape[0]) * self.__los_ratio / (1 - self.__los_ratio))
            chosen_los_idx = random.choices(los_idx, k=num_los)
            chosen_idx = chosen_los_idx + nlos_idx.tolist()
            random.shuffle(chosen_idx)
            return chosen_idx
