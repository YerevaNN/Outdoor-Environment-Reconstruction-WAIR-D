import logging
import os

import numpy as np

from src.datamodules.datasets import WAIRDDatasetImages
from src.utils import EpochCounter

log = logging.getLogger(__name__)


class WAIRDDatasetImagesSequence(WAIRDDatasetImages):
    
    def __init__(
        self, image_data_path: str, sequence_data_path: str, scenario: str, scenario2_path: str,
        split: str, output_kernel_size: int, kernel_size_decay: int, epoch_counter: EpochCounter,
        use_channels_img: list[int], use_channels_seq: list[int], n_links: int, los_ratio: float,
        input_kernel_size: float, no_supervision_image: bool
    ):
        super().__init__(
            image_data_path, scenario, scenario2_path, split,
            output_kernel_size, kernel_size_decay, epoch_counter,
            use_channels_img, los_ratio, input_kernel_size, no_supervision_image
        )
        self.__sequence_data_path = os.path.join(sequence_data_path, scenario)
        self.__n_links = n_links
        self.__use_channels_seq = use_channels_seq
    
    def __getitem__(self, data_idx: int):
        if self._split == "train" and self._los_ratio < 1:
            pair_idx = self._chosen_idx[data_idx]
        else:
            pair_idx = data_idx
        
        input_img, ue_loc_img, ue_location, img_size, is_los = super().__getitem__(data_idx)
        
        environment_idx: int = pair_idx // self._num_pairs_per_env
        local_pair_idx = pair_idx % self._num_pairs_per_env
        bs_idx = local_pair_idx // self._num_ues_per_env
        ue_idx = local_pair_idx % self._num_ues_per_env
        environment: str = self._environments[environment_idx]
        env_path = os.path.join(self.__sequence_data_path, environment)
        
        pair_path = os.path.join(env_path, f"bs{bs_idx}_ue{ue_idx}")
        pair_data = np.load(os.path.join(pair_path, "pairs_data.npz"), allow_pickle=True)
        
        locations = pair_data["locations"].item()
        bs_location = locations['bs'].astype(np.float32)[:2].astype(np.float32)
        
        sequence = pair_data["sequence"].astype(np.float32)
        sequence = np.concatenate([np.tile(bs_location, sequence.shape[0]).reshape(-1, 2), sequence], axis=1)
        
        pad_len = max(0, self.__n_links - sequence.shape[0])
        mask = np.array([0] * min(sequence.shape[0], self.__n_links) + [1] * pad_len, dtype=np.bool_)
        pad = np.zeros((pad_len, sequence.shape[1]))
        sequence = np.vstack([sequence, pad])
        sequence = sequence[: self.__n_links]
        if self.__use_channels_seq is not None:
            sequence = sequence[:, self.__use_channels_seq]
        sequence = sequence.astype(np.float32)
        
        return input_img, sequence, mask, ue_loc_img, ue_location, img_size, is_los
