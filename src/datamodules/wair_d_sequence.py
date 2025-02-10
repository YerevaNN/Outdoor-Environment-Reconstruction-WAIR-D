import torch

from src.datamodules.datasets import WAIRDDatasetSequence
from src.datamodules.wair_d_base import WAIRDBaseDatamodule


class WAIRDSequenceDatamodule(WAIRDBaseDatamodule):
    
    def __init__(
        self, data_path: str, scenario: str, scenario2_path: str,
        batch_size: int, num_workers: int, use_channels: list[int], n_links: int, los_ratio: float,
        multi_gpu: bool = False, *args, **kwargs
    ):
        self.__data_path = data_path
        self.__scenario = scenario
        self.__scenario2_path = scenario2_path
        self.__use_channels = use_channels
        self.__n_links = n_links
        self.__los_ratio = los_ratio
        
        super().__init__(batch_size=batch_size, num_workers=num_workers, multi_gpu=multi_gpu)
        
        if n_links is not None:
            self.collate_fn = None
    
    @staticmethod
    def collate_fn(items):
        sequences, masks, ue_locations, img_sizes, is_loss = [], [], [], [], []
        
        max_len = 0
        for item in items:
            sequence, mask, ue_location, img_size, is_los = item
            
            l = sequence.shape[0]
            if l > max_len:
                max_len = l
        
        for item in items:
            sequence, mask, ue_location, img_size, is_los = item
            
            pad_len = max(0, max_len - sequence.shape[0])
            mask = torch.cat([torch.zeros(mask.shape[0], dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)])
            pad = torch.zeros((pad_len, sequence.shape[1]))
            sequence = torch.cat([torch.Tensor(sequence), pad])
            
            sequences.append(sequence)
            masks.append(mask)
            ue_locations.append(torch.tensor(ue_location))
            img_sizes.append(torch.tensor(img_size))
            is_loss.append(torch.tensor(is_los))
        
        sequences = torch.stack(sequences)
        masks = torch.stack(masks)
        ue_locations = torch.stack(ue_locations)
        img_sizes = torch.stack(img_sizes)
        is_loss = torch.stack(is_loss)
        
        return sequences, masks, ue_locations, img_sizes, is_loss
    
    def prepare_data(self) -> None:
        self._train_set = WAIRDDatasetSequence(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="train",
            use_channels=self.__use_channels, n_links=self.__n_links, los_ratio=self.__los_ratio
        )
        self._val_set = WAIRDDatasetSequence(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="val",
            use_channels=self.__use_channels, n_links=self.__n_links, los_ratio=self.__los_ratio
        )
        self._test_set = WAIRDDatasetSequence(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="test",
            use_channels=self.__use_channels, n_links=self.__n_links, los_ratio=self.__los_ratio
        )
