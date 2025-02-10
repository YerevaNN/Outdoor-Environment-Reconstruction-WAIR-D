from src.datamodules.datasets import WAIRDDatasetImages
from src.datamodules.wair_d_base import WAIRDBaseDatamodule
from src.utils import EpochCounter


class WAIRDImagesDatamodule(WAIRDBaseDatamodule):
    
    def __init__(
        self, data_path: str, scenario: str, scenario2_path: str,
        batch_size: int, num_workers: int,
        output_kernel_size: int, kernel_size_decay: int, epoch_counter: EpochCounter,
        use_channels: list[int], los_ratio: float, input_kernel_size: float, no_supervision_image: bool,
        multi_gpu: bool = False, *args, **kwargs
    ):
        self.__data_path = data_path
        self.__scenario = scenario
        self.__scenario2_path = scenario2_path
        self.__output_kernel_size = output_kernel_size
        self.__kernel_size_decay = kernel_size_decay
        self.__epoch_counter = epoch_counter
        self.__use_channels = use_channels
        self.__los_ratio = los_ratio
        self.__input_kernel_size = input_kernel_size
        self.__no_supervision_image = no_supervision_image
        super().__init__(batch_size=batch_size, num_workers=num_workers, multi_gpu=multi_gpu)
    
    def prepare_data(self) -> None:
        self._train_set = WAIRDDatasetImages(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="train",
            output_kernel_size=self.__output_kernel_size, kernel_size_decay=self.__kernel_size_decay,
            epoch_counter=self.__epoch_counter, use_channels=self.__use_channels, los_ratio=self.__los_ratio,
            input_kernel_size=self.__input_kernel_size, no_supervision_image=self.__no_supervision_image
        )
        self._val_set = WAIRDDatasetImages(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="val",
            output_kernel_size=self.__output_kernel_size, kernel_size_decay=self.__kernel_size_decay,
            epoch_counter=self.__epoch_counter, use_channels=self.__use_channels, los_ratio=self.__los_ratio,
            input_kernel_size=self.__input_kernel_size, no_supervision_image=self.__no_supervision_image
        )
        self._test_set = WAIRDDatasetImages(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="test",
            output_kernel_size=self.__output_kernel_size, kernel_size_decay=self.__kernel_size_decay,
            epoch_counter=self.__epoch_counter, use_channels=self.__use_channels, los_ratio=self.__los_ratio,
            input_kernel_size=self.__input_kernel_size, no_supervision_image=self.__no_supervision_image
        )
