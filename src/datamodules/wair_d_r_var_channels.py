from src.datamodules.datasets import WAIRDDatasetRVarChannels
from src.datamodules.wair_d_base import WAIRDBaseDatamodule


class WAIRDRVarChannelsDatamodule(WAIRDBaseDatamodule):
    
    def __init__(
        self, data_path: str, scenario: str, scenario2_path: str, batch_size: int, num_workers: int,
        channels_range: list[int], aoa_aod: list[int], multi_gpu: bool = False, *args, **kwargs
    ):
        self.__data_path = data_path
        self.__scenario = scenario
        self.__scenario2_path = scenario2_path
        
        self.__aoa_aod = aoa_aod
        self.__channels_range = channels_range
        super().__init__(batch_size=batch_size, num_workers=num_workers, multi_gpu=multi_gpu)
    
    def prepare_data(self) -> None:
        self._train_set = WAIRDDatasetRVarChannels(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="train",
            channels_range=self.__channels_range, aoa_aod=self.__aoa_aod
        )
        self._val_set = WAIRDDatasetRVarChannels(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="val",
            channels_range=self.__channels_range, aoa_aod=self.__aoa_aod
        )
        self._test_set = WAIRDDatasetRVarChannels(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="test",
            channels_range=self.__channels_range, aoa_aod=self.__aoa_aod
        )
