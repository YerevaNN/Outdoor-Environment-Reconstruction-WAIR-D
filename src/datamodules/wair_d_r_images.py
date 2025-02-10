from src.datamodules.datasets import WAIRDDatasetRImages
from src.datamodules.wair_d_base import WAIRDBaseDatamodule


class WAIRDRImagesDatamodule(WAIRDBaseDatamodule):
    
    def __init__(
        self, data_path: str, scenario: str, scenario2_path: str, batch_size: int, num_workers: int,
        num_bs: tuple[int, int], num_ue: tuple[int, int], aoa_aod: list[int],
        random_sample: int, random_crop: int, middle_crop: int, random_flip: bool, random_rotate: bool,
        multi_gpu: bool = False,
        *args, **kwargs
    ):
        self.__data_path = data_path
        self.__scenario = scenario
        self.__scenario2_path = scenario2_path
        self.__num_bs = num_bs
        self.__num_ue = num_ue
        self.__random_sample = random_sample
        self.__random_crop = random_crop
        self.__middle_crop = middle_crop
        self.__random_flip = random_flip
        self.__random_rotate = random_rotate
        self.__aoa_aod = aoa_aod
        super().__init__(batch_size=batch_size, num_workers=num_workers, multi_gpu=multi_gpu)
    
    def prepare_data(self) -> None:
        self._train_set = WAIRDDatasetRImages(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="train",
            num_bs=self.__num_bs, num_ue=self.__num_ue, aoa_aod=self.__aoa_aod,
            random_sample=self.__random_sample, random_crop=self.__random_crop, middle_crop=self.__middle_crop,
            random_flip=self.__random_flip, random_rotate=self.__random_rotate
        )
        self._val_set = WAIRDDatasetRImages(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="val",
            num_bs=self.__num_bs, num_ue=self.__num_ue, aoa_aod=self.__aoa_aod,
            random_sample=self.__random_sample, random_crop=self.__random_crop, middle_crop=self.__middle_crop,
            random_flip=self.__random_flip, random_rotate=self.__random_rotate
        )
        self._test_set = WAIRDDatasetRImages(
            data_path=self.__data_path, scenario=self.__scenario, scenario2_path=self.__scenario2_path, split="test",
            num_bs=self.__num_bs, num_ue=self.__num_ue, aoa_aod=self.__aoa_aod,
            random_sample=self.__random_sample, random_crop=self.__random_crop, middle_crop=self.__middle_crop,
            random_flip=self.__random_flip, random_rotate=self.__random_rotate
        )
