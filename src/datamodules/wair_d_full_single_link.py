from src.datamodules.datasets import WAIRDDatasetFullSingleLink
from src.datamodules.wair_d_base import WAIRDBaseDatamodule


class WAIRDFullSingleLinkDatamodule(WAIRDBaseDatamodule):
    
    def __init__(
        self, data_path: str, scenario: str, batch_size: int, num_workers: int, multi_gpu: bool = False,
        *args, **kwargs
    ):
        self.__data_path = data_path
        self.__scenario = scenario
        super().__init__(batch_size=batch_size, num_workers=num_workers, multi_gpu=multi_gpu)
    
    def prepare_data(self) -> None:
        self._train_set = WAIRDDatasetFullSingleLink(
            data_path=self.__data_path, scenario=self.__scenario, split="train"
        )
        self._val_set = WAIRDDatasetFullSingleLink(data_path=self.__data_path, scenario=self.__scenario, split="val")
        self._test_set = WAIRDDatasetFullSingleLink(data_path=self.__data_path, scenario=self.__scenario, split="test")
