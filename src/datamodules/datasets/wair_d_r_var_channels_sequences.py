import logging
import random

from torch.utils.data import Dataset

from src.datamodules.datasets.wair_d_r_sequences import WAIRDDatasetRSequences
from src.datamodules.datasets.wair_d_r_var_channels import WAIRDDatasetRVarChannels

log = logging.getLogger(__name__)


class WAIRDDatasetRVarChannelsSequences(Dataset):
    
    def __init__(
        self, images_data_path: str, sequences_data_path: str, scenario: str, scenario2_path: str, split: str,
        channels_range: list[int], aoa_aod: list[int], use_channels: list[int], n_links: int
    ):
        super().__init__()
        
        if scenario != "scenario_1":
            raise NotImplementedError(f'There is only an implementation for scenario_1. Given {scenario}')
        
        if split != "train" and split != "val" and split != "test":
            raise NotImplementedError(f'There is only an implementation for train and test. Given {split}')
        
        self.__wair_d_dataset_r_var_channels = WAIRDDatasetRVarChannels(
            data_path=images_data_path, scenario=scenario, scenario2_path=scenario2_path, split=split,
            channels_range=channels_range, aoa_aod=aoa_aod
        )
        self.__wair_d_dataset_r_sequences = WAIRDDatasetRSequences(
            data_path=sequences_data_path, scenario=scenario, scenario2_path=scenario2_path, split=split,
            use_channels=use_channels, n_links=n_links, n_tokens=channels_range[1]
        )
        
        random.seed(42)
        self._val_test_channels = random.sample(range(150), channels_range[1])
        self.__wair_d_dataset_r_var_channels._val_test_channels = self._val_test_channels
        self.__wair_d_dataset_r_sequences._val_test_tokens = self._val_test_channels
        self.__channels_range = channels_range
        
        self.__split: str = split
        
        self.__num_train_envs_count = 10000 - 500 - 100 - 1 - 96
        self.__num_val_envs_count = 500 + 96
        self.__num_test_envs_count = 100
        
        self._environments: list[str] = self.__wair_d_dataset_r_var_channels._environments
    
    def __getitem__(self, idx: int):
        if self.__split == "train":
            n_channels = random.randint(*self.__channels_range)
            train_channels = random.sample(range(150), n_channels)
            self.__wair_d_dataset_r_var_channels._train_channels = train_channels
            self.__wair_d_dataset_r_sequences._train_tokens = train_channels
        
        input_img, map_img = self.__wair_d_dataset_r_var_channels[idx]
        sequences, img_size = self.__wair_d_dataset_r_sequences[idx]
        
        return input_img, map_img, sequences, img_size
    
    def __len__(self):
        if self.__split == "train":
            return self.__num_train_envs_count
        elif self.__split == "val":
            return self.__num_val_envs_count
        else:
            return self.__num_test_envs_count
