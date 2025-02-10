import logging
import os
import random

import numpy as np
import scipy
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class WAIRDDatasetRNChannels(Dataset):
    
    def __init__(
        self, data_path: str, scenario: str, scenario2_path: str, split: str,
        n_channels: int, aoa_aod: list[int],
    ):
        super().__init__()
        
        if scenario != "scenario_1":
            raise NotImplementedError(f'There is only an implementation for scenario_1. Given {scenario}')
        
        if split != "train" and split != "val" and split != "test":
            raise NotImplementedError(f'There is only an implementation for train and test. Given {split}')
        
        self.__scenario_path: str = os.path.join(data_path, scenario)
        self.__scenario2_path = scenario2_path
        self.__split: str = split
        
        self.__aoa_aod = aoa_aod
        self.__n_channels = n_channels
        
        random.seed(42)
        self._train_channels = None
        self._val_test_channels = random.sample(range(150), n_channels)
        
        self.__num_envs = 10000 - 1
        
        self.__num_train_envs_count = 10000 - 500 - 100 - 1 - 96
        self.__num_val_envs_count = 500 + 96
        self.__num_test_envs_count = 100
        
        self._environments: list[str] = self.__prepare_environments()
    
    def __getitem__(self, idx: int):
        if self.__split == "train":
            channels_to_use = self._train_channels or random.sample(range(150), self.__n_channels)
        else:
            channels_to_use = self._val_test_channels
        
        channels_to_use = [2 * c + a for c in channels_to_use for a in self.__aoa_aod]
        environment: str = self._environments[idx]
        env_path = os.path.join(self.__scenario_path, environment)
        
        map_img = list(np.load(os.path.join(env_path, "environment.npz")).values())[0].astype(np.float32)
        map_img = map_img[np.newaxis]
        
        input_img = scipy.sparse.load_npz(os.path.join(env_path, "input_img.npz")).toarray()
        input_img = input_img.reshape(-1, input_img.shape[-1], input_img.shape[-1])
        input_img = input_img[channels_to_use]
        
        return input_img, map_img
    
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
