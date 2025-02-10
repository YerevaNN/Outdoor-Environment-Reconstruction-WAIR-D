import logging
import os
import random
from itertools import combinations

import numpy as np
import scipy
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class WAIRDDatasetRImages(Dataset):
    
    def __init__(
        self, data_path: str, scenario: str, scenario2_path: str, split: str,
        num_bs: tuple[int, int], num_ue: tuple[int, int], aoa_aod: list[int],
        random_sample: int, random_crop: int, middle_crop: int, random_flip: bool, random_rotate: bool
    ):
        super().__init__()
        
        if scenario != "scenario_1":
            raise NotImplementedError(f'There is only an implementation for scenario_1. Given {scenario}')
        
        if split != "train" and split != "val" and split != "test":
            raise NotImplementedError(f'There is only an implementation for train and test. Given {split}')
        
        self.__scenario_path: str = os.path.join(data_path, scenario)
        self.__scenario2_path = scenario2_path
        self.__split: str = split
        self.__num_bs = num_bs
        self.__num_ue = num_ue
        self.__random_sample = random_sample
        self.__random_crop = random_crop
        self.__middle_crop = middle_crop
        self.__random_flip = random_flip
        self.__random_rotate = random_rotate
        if not self.__random_sample:
            self.__bs_exclude = list(combinations(range(5), 5 - self.__num_bs[0]))
            self.__ue_combinations = list(combinations(range(30), self.__num_ue[0]))
        self.__aoa_aod = aoa_aod
        
        self.__num_envs = 10000 - 1
        
        self.__num_train_envs_count = 10000 - 500 - 100 - 1 - 96
        self.__num_val_envs_count = 500 + 96
        self.__num_test_envs_count = 100
        
        self.__environments: list[str] = self.__prepare_environments()
    
    def __getitem__(self, idx: int):
        if self.__split != "train":
            pairs_count = 1
            ue_combination = range(30)
            bs_exclude = []
            bss = range(5)
        else:
            pairs_count = self.__random_sample or (len(self.__bs_exclude) * len(self.__ue_combinations))
            if not self.__random_sample:
                combination_idx = idx % pairs_count
                bs_exclude_idx = combination_idx // len(self.__ue_combinations)
                ue_combination_idx = combination_idx % len(self.__ue_combinations)
                
                bss = range(5)
                bs_exclude = self.__bs_exclude[bs_exclude_idx]
                ue_combination = self.__ue_combinations[ue_combination_idx]
            else:
                num_bs = random.randint(*self.__num_bs)
                num_ue = random.randint(*self.__num_ue)
                bss = random.sample(range(5), 5)
                bs_exclude = random.sample(range(5), 5 - num_bs)
                ue_combination = random.sample(range(30), num_ue)
        
        channels_to_use = [[bs * 60 + 2 * ue + a for ue in ue_combination] for bs in bss for a in self.__aoa_aod]
        environment_idx = idx // pairs_count
        environment: str = self.__environments[environment_idx]
        env_path = os.path.join(self.__scenario_path, environment)
        
        map_img = list(np.load(os.path.join(env_path, "environment.npz")).values())[0].astype(np.float32)
        map_img = map_img[np.newaxis]
        
        input_img = scipy.sparse.load_npz(os.path.join(env_path, "input_img.npz")).toarray()
        input_img = input_img.reshape(-1, input_img.shape[-1], input_img.shape[-1])
        # filtering number of shortest paths
        # if self.__num_shortest_paths:
        #     for input_channel in input_img:
        #         channel_uniques = np.unique(input_channel)
        #         n_th_biggest = channel_uniques[-self.__num_shortest_paths] if \
        #             self.__num_shortest_paths <= len(channel_uniques) else 0
        #         input_channel[input_channel < n_th_biggest] = 0
        # combining channels corresponding to the same base station and angle of arrival/departure
        input_img = input_img[channels_to_use].max(axis=1)
        input_img[[bs + a for bs in bs_exclude for a in self.__aoa_aod]] = 0
        
        crop = self.__random_crop or self.__middle_crop
        if self.__split == "train":
            if self.__random_flip:
                flip_axis = random.choice([(), 1, 2, (1, 2)])
                input_img = np.flip(input_img, axis=flip_axis)
                map_img = np.flip(map_img, axis=flip_axis)
            
            if self.__random_rotate:
                rotation_angle = random.uniform(-180, 180)
                input_img = scipy.ndimage.rotate(input_img, rotation_angle, axes=(1, 2), reshape=False, order=0)
                map_img = scipy.ndimage.rotate(map_img, rotation_angle, axes=(1, 2), reshape=False, order=0)
            
            if crop:
                if self.__random_crop:
                    crop_x = random.randint(0, input_img.shape[-1] - crop)
                    crop_y = random.randint(0, input_img.shape[-1] - crop)
                else:
                    crop_x = crop_y = (input_img.shape[-1] - crop) // 2
                input_img = input_img[:, crop_x: crop_x + crop, crop_y: crop_y + crop]
                map_img = map_img[:, crop_x: crop_x + crop, crop_y: crop_y + crop]
        elif crop:
            crop_x = crop_y = (input_img.shape[-1] - crop) // 2
            input_img = input_img[:, crop_x: crop_x + crop, crop_y: crop_y + crop]
            map_img = map_img[:, crop_x: crop_x + crop, crop_y: crop_y + crop]
        
        return input_img, map_img
    
    def __len__(self):
        pairs_count = 1
        if self.__split == "train":
            env_count = self.__num_train_envs_count
            pairs_count = self.__random_sample or (len(self.__bs_exclude) * len(self.__ue_combinations))
        elif self.__split == "val":
            env_count = self.__num_val_envs_count
        else:
            env_count = self.__num_test_envs_count
        
        return env_count * pairs_count
    
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
