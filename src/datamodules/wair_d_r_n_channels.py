from src.datamodules.datasets import WAIRDDatasetRNChannels
from src.datamodules.wair_d_base import WAIRDBaseDatamodule


class WAIRDRNChannelsDatamodule(WAIRDBaseDatamodule):
    
    def __init__(
        self, data_path: str, scenario: str, scenario2_path: str, batch_size: int, num_workers: int,
        n_channels: int, aoa_aod: list[int], multi_gpu: bool = False, *args, **kwargs
    ):
        self.__data_path = data_path
        self.__scenario = scenario
        self.__scenario2_path = scenario2_path
        
        self.__aoa_aod = aoa_aod
        self.__n_channels = n_channels
        super().__init__(batch_size=batch_size, num_workers=num_workers, multi_gpu=multi_gpu)
    
    def prepare_data(self) -> None:
        self._train_set = WAIRDDatasetRNChannels(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="train",
            n_channels=self.__n_channels, aoa_aod=self.__aoa_aod
        )
        self._val_set = WAIRDDatasetRNChannels(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="val",
            n_channels=self.__n_channels, aoa_aod=self.__aoa_aod
        )
        self._test_set = WAIRDDatasetRNChannels(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="test",
            n_channels=self.__n_channels, aoa_aod=self.__aoa_aod
        )
