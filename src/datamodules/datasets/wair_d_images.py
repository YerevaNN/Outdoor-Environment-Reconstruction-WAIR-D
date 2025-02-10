import logging
import os
import random

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import EpochCounter

log = logging.getLogger(__name__)


class WAIRDDatasetImages(Dataset):
    
    def __init__(
        self, data_path: str, scenario: str, scenario2_path: str, split: str,
        output_kernel_size: int, kernel_size_decay: int, epoch_counter: EpochCounter,
        use_channels: list[int], los_ratio: float, input_kernel_size: float, no_supervision_image: bool
    ):
        super().__init__()
        
        if scenario != "scenario_1":
            raise NotImplementedError(f'There is only an implementation for scenario_1. Given {scenario}')
        
        if split != "train" and split != "val" and split != "test":
            raise NotImplementedError(f'There is only an implementation for train and test. Given {split}')
        
        self.__scenario_path: str = os.path.join(data_path, scenario)
        self.__scenario2_path: str = scenario2_path
        self._split: str = split
        self.__output_kernel_size: int = output_kernel_size
        self.__kernel_size_decay = kernel_size_decay
        self.__epoch_counter = epoch_counter
        self._use_channels = use_channels
        self._los_ratio = los_ratio
        self.__input_kernel_size = input_kernel_size
        self.__no_supervision_image = no_supervision_image
        
        self.__num_envs = 10000 - 1
        
        self.__num_train_envs_count = 10000 - 500 - 100 - 1 - 96
        self.__num_val_envs_count = 500 + 96
        self.__num_test_envs_count = 100
        
        self.__num_bss_per_env = 5
        self._num_ues_per_env = 30
        self._num_pairs_per_env = self.__num_bss_per_env * self._num_ues_per_env
        
        self._environments: list[str] = self.__prepare_environments()
        self._chosen_idx = self.__subsample_los()
    
    def __getitem__(self, data_idx: int):
        if self._split == "train" and self._los_ratio < 1:
            pair_idx = self._chosen_idx[data_idx]
        else:
            pair_idx = data_idx
        
        environment_idx: int = pair_idx // self._num_pairs_per_env
        
        local_pair_idx = pair_idx % self._num_pairs_per_env
        bs_idx = local_pair_idx // self._num_ues_per_env
        ue_idx = local_pair_idx % self._num_ues_per_env
        
        environment: str = self._environments[environment_idx]
        
        env_path = os.path.join(self.__scenario_path, environment)
        metadata = dict(np.load(os.path.join(env_path, "metadata.npz")))
        
        pair_path = os.path.join(env_path, f"bs{bs_idx}_ue{ue_idx}")
        pair_data_path = os.path.join(pair_path, "data.npz")
        pair_data = dict(np.load(pair_data_path, allow_pickle=True))
        
        input_img = list(np.load(os.path.join(pair_path, "input_img.npz")).values())[0].astype(np.float32)
        
        locations = pair_data["locations"].item()
        ue_location = locations['ue'].astype(np.float32)[:2].astype(np.float32) * max(input_img.shape)
        
        input_img = resize(input_img, (3, 224, 224))
        
        if self.__no_supervision_image:
            ue_loc_img = float("NaN")
        else:
            ue_loc_img = list(np.load(os.path.join(pair_path, "ue_loc_img.npz")).values())[0].astype(np.float32)
            ue_loc_img = resize(ue_loc_img, (1, 224, 224))
            output_kernel_size = self.__output_kernel_size / (self.__kernel_size_decay ** self.__epoch_counter.count)
            ue_loc_img = gaussian_filter(ue_loc_img, output_kernel_size)
            ue_loc_img_max = ue_loc_img.max()
            if ue_loc_img_max != 0:
                ue_loc_img /= ue_loc_img_max
        
        if self.__input_kernel_size is not None:
            input_img = self.__remove_rays(input_img)
        
        if self._use_channels is not None:
            input_img = input_img[self._use_channels]
        
        return (
            input_img,
            ue_loc_img,
            ue_location,
            metadata["img_size"] / 2,
            pair_data["is_los"]
        )
    
    def __len__(self):
        if self._split == "train":
            if self._los_ratio < 1:
                return len(self._chosen_idx)
            env_count = self.__num_train_envs_count
        elif self._split == "val":
            env_count = self.__num_val_envs_count
        else:
            env_count = self.__num_test_envs_count
        
        return env_count * self.__num_bss_per_env * self._num_ues_per_env
    
    def __prepare_environments(self) -> list[str]:
        environments = os.listdir(self.__scenario_path)
        environments = sorted(filter(str.isnumeric, environments))
        scenario2_environments = set(filter(str.isnumeric, os.listdir(self.__scenario2_path)))
        
        if self._split == "train":
            return sorted(set(environments[:900] + environments[1000: 9499]) - scenario2_environments)
        elif self._split == "val":
            return sorted(set(environments[9499:]) | scenario2_environments)
        else:
            return environments[900:1000]
    
    def __subsample_los(self) -> list[int]:
        if self._split == "train" and self._los_ratio < 1:
            log.info("Subsampling LOS data")
            is_los_list = []
            for env in tqdm(self._environments):
                env_path = os.path.join(self.__scenario_path, env)
                for bs_idx in range(self.__num_bss_per_env):
                    for ue_idx in range(self._num_ues_per_env):
                        pair_path = os.path.join(env_path, f"bs{bs_idx}_ue{ue_idx}")
                        pair_data = np.load(os.path.join(pair_path, "data.npz"), allow_pickle=True)
                        is_los = pair_data["is_los"]
                        is_los_list.append(is_los)
            is_los_list = np.array(is_los_list)
            los_idx = np.where(is_los_list)[0]
            nlos_idx = np.where(~is_los_list)[0]
            # num_los = int(los_idx.shape[0] * los_ratio)
            num_los = int((is_los_list.shape[0] - los_idx.shape[0]) * self._los_ratio / (1 - self._los_ratio))
            chosen_los_idx = random.choices(los_idx, k=num_los)
            chosen_idx = chosen_los_idx + nlos_idx.tolist()
            random.shuffle(chosen_idx)
            return chosen_idx
    
    def __remove_rays(self, input_img):
        eps = 1e-7
        im = input_img[1]
        im[im == 0] = -10
        im = gaussian_filter(im, self.__input_kernel_size)
        im[im < im.max()] = -10
        im = gaussian_filter(im + 10, self.__input_kernel_size)
        im /= max(im.max(), eps) if im.max() > 0 else min(im.max(), -eps)
        return np.stack([input_img[0], im])
