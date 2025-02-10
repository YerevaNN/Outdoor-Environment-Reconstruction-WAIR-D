from src.datamodules.datasets import WAIRDDatasetRNChannelsSequences
from src.datamodules.wair_d_base import WAIRDBaseDatamodule


class WAIRDRNChannelsSequencesDatamodule(WAIRDBaseDatamodule):
    
    def __init__(
        self, images_data_path: str, sequences_data_path: str, scenario: str, scenario2_path: str,
        batch_size: int, num_workers: int,
        n_channels: int, aoa_aod: list[int], use_channels: list[int], n_links: int, multi_gpu: bool = False,
        *args, **kwargs
    ):
        self.__images_data_path = images_data_path
        self.__sequences_data_path = sequences_data_path
        self.__scenario = scenario
        self.__scenario2_path = scenario2_path
        
        self.__n_channels = n_channels
        self.__aoa_aod = aoa_aod
        self.__use_channels = use_channels
        self.__n_links = n_links
        super().__init__(batch_size=batch_size, num_workers=num_workers, multi_gpu=multi_gpu)
    
    def prepare_data(self) -> None:
        self._train_set = WAIRDDatasetRNChannelsSequences(
            images_data_path=self.__images_data_path, sequences_data_path=self.__sequences_data_path,
            scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="train",
            n_channels=self.__n_channels, aoa_aod=self.__aoa_aod,
            use_channels=self.__use_channels, n_links=self.__n_links
        )
        self._val_set = WAIRDDatasetRNChannelsSequences(
            images_data_path=self.__images_data_path, sequences_data_path=self.__sequences_data_path,
            scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="val",
            n_channels=self.__n_channels, aoa_aod=self.__aoa_aod,
            use_channels=self.__use_channels, n_links=self.__n_links
        )
        self._test_set = WAIRDDatasetRNChannelsSequences(
            images_data_path=self.__images_data_path, sequences_data_path=self.__sequences_data_path,
            scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="test",
            n_channels=self.__n_channels, aoa_aod=self.__aoa_aod,
            use_channels=self.__use_channels, n_links=self.__n_links
        )
