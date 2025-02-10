import logging
import os
import random
from itertools import zip_longest

import numpy as np
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class WAIRDDatasetRSequences(Dataset):
    
    def __init__(
        self, data_path: str, scenario: str, scenario2_path: str, split: str,
        use_channels: list[int], n_links: int, n_tokens: int
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
        self.__n_tokens = n_tokens
        
        random.seed(42)
        self._train_tokens = None
        self._val_test_tokens = random.sample(range(150), n_tokens)
        
        self.__num_envs = 10000 - 1
        
        self.__num_train_envs_count = 10000 - 500 - 100 - 1 - 96
        self.__num_val_envs_count = 500 + 96
        self.__num_test_envs_count = 100
        
        self.__num_bss_per_env = 5
        self.__num_ues_per_env = 30
        self.__num_pairs_per_env = self.__num_bss_per_env * self.__num_ues_per_env
        
        self.__environments: list[str] = self.__prepare_environments()
    
    def __getitem__(self, environment_idx: int):
        if self.__split == "train":
            tokens_to_use = self._train_tokens or random.sample(range(150), self.__n_tokens)
        else:
            tokens_to_use = self._val_test_tokens
        
        environment: str = self.__environments[environment_idx]
        env_path = os.path.join(self.__scenario_path, environment)
        
        ue_locations = []
        bs_locations = []
        is_los_list = []
        sequences = []
        
        for t in tokens_to_use:
            bs_idx = t // self.__num_ues_per_env
            ue_idx = t % self.__num_ues_per_env
            pair_path = os.path.join(env_path, f"bs{bs_idx}_ue{ue_idx}")
            pair_data = np.load(os.path.join(pair_path, "pairs_data.npz"), allow_pickle=True)
            
            img_size = pair_data["metadata"].item()["img_size"]
            locations = pair_data["locations"].item()
            ue_location = locations['ue'].astype(np.float32)[:2]
            bs_location = locations['bs'].astype(np.float32)[:2]
            is_los = pair_data["is_los"]
            sequence = pair_data["sequence"].astype(np.float32)
            
            ue_locations.append(ue_location)
            bs_locations.append(bs_location)
            is_los_list.append(is_los)
            sequences.append(sequence[:self.__n_links, self.__use_channels].flatten())
        
        ue_locations = np.array(ue_locations)
        bs_locations = np.array(bs_locations)
        is_los_list = np.array(is_los_list)
        sequences = np.array(list(zip_longest(*sequences, fillvalue=0))).T
        pad = np.zeros((sequences.shape[0], self.__n_links * len(self.__use_channels) - sequences.shape[1]))
        sequences = np.concatenate([ue_locations, bs_locations, is_los_list[:, np.newaxis], sequences, pad], axis=1)
        sequences = sequences.astype(np.float32)
        
        return sequences, img_size / 2
    
    def __len__(self):
        if self.__split == "train":
            return self.__num_train_envs_count
        elif self.__split == "val":
            return self.__num_val_envs_count
        else:
            return self.__num_test_envs_count
    
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
