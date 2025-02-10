import torch

from src.datamodules.datasets import WAIRDDatasetRVarChannelsSequences
from src.datamodules.wair_d_base import WAIRDBaseDatamodule


class WAIRDRVarChannelsSequencesDatamodule(WAIRDBaseDatamodule):
    
    def __init__(
        self, images_data_path: str, sequences_data_path: str, scenario: str, scenario2_path: str,
        batch_size: int, num_workers: int,
        channels_range: list[int], aoa_aod: list[int], use_channels: list[int], n_links: int, multi_gpu: bool = False,
        *args, **kwargs
    ):
        self.__images_data_path = images_data_path
        self.__sequences_data_path = sequences_data_path
        self.__scenario = scenario
        self.__scenario2_path = scenario2_path
        
        self.__channels_range = channels_range
        self.__aoa_aod = aoa_aod
        self.__use_channels = use_channels
        self.__n_links = n_links
        super().__init__(batch_size=batch_size, num_workers=num_workers, multi_gpu=multi_gpu)
    
    @staticmethod
    def collate_fn(items):
        input_images, map_images, sequences, img_sizes = [], [], [], []
        
        max_len = 0
        for item in items:
            input_img, map_img, sequence, img_size = item
            
            l = sequence.shape[0]
            if l > max_len:
                max_len = l
        
        for item in items:
            input_img, map_img, sequence, img_size = item
            
            pad_len = max(0, max_len - sequence.shape[0])
            pad = torch.zeros((pad_len, sequence.shape[1]))
            sequence = torch.cat([torch.Tensor(sequence), pad])
            
            sequences.append(sequence)
            input_images.append(torch.tensor(input_img))
            img_sizes.append(torch.tensor(img_size))
            map_images.append(torch.Tensor(map_img))
        
        sequences = torch.stack(sequences)
        input_images = torch.stack(input_images)
        img_sizes = torch.stack(img_sizes)
        map_images = torch.stack(map_images)
        
        return input_images, map_images, sequences, img_sizes
    
    def prepare_data(self) -> None:
        self._train_set = WAIRDDatasetRVarChannelsSequences(
            images_data_path=self.__images_data_path, sequences_data_path=self.__sequences_data_path,
            scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="train",
            channels_range=self.__channels_range, aoa_aod=self.__aoa_aod,
            use_channels=self.__use_channels, n_links=self.__n_links
        )
        self._val_set = WAIRDDatasetRVarChannelsSequences(
            images_data_path=self.__images_data_path, sequences_data_path=self.__sequences_data_path,
            scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="val",
            channels_range=self.__channels_range, aoa_aod=self.__aoa_aod,
            use_channels=self.__use_channels, n_links=self.__n_links
        )
        self._test_set = WAIRDDatasetRVarChannelsSequences(
            images_data_path=self.__images_data_path, sequences_data_path=self.__sequences_data_path,
            scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="test",
            channels_range=self.__channels_range, aoa_aod=self.__aoa_aod,
            use_channels=self.__use_channels, n_links=self.__n_links
        )
