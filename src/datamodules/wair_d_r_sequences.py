from src.datamodules.datasets import WAIRDDatasetRSequences
from src.datamodules.wair_d_base import WAIRDBaseDatamodule


class WAIRDRSequencesDatamodule(WAIRDBaseDatamodule):
    
    def __init__(
        self, data_path: str, scenario: str, scenario2_path: str,
        batch_size: int, num_workers: int, use_channels: list[int], n_links: int, n_tokens: int,
        multi_gpu: bool = False, *args, **kwargs
    ):
        self.__data_path = data_path
        self.__scenario = scenario
        self.__scenario2_path = scenario2_path
        self.__use_channels = use_channels
        self.__n_links = n_links
        self.__n_tokens = n_tokens
        super().__init__(batch_size=batch_size, num_workers=num_workers, multi_gpu=multi_gpu)
    
    def prepare_data(self) -> None:
        self._train_set = WAIRDDatasetRSequences(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="train",
            use_channels=self.__use_channels, n_links=self.__n_links, n_tokens=self.__n_tokens
        )
        self._val_set = WAIRDDatasetRSequences(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="val",
            use_channels=self.__use_channels, n_links=self.__n_links, n_tokens=self.__n_tokens
        )
        self._test_set = WAIRDDatasetRSequences(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="test",
            use_channels=self.__use_channels, n_links=self.__n_links, n_tokens=self.__n_tokens
        )
