import torch

from src.datamodules.datasets import WAIRDDatasetFullAllLinks
from src.datamodules.wair_d_base import WAIRDBaseDatamodule


class WAIRDFullAllLinksDatamodule(WAIRDBaseDatamodule):
    
    def __init__(
        self, data_path: str, scenario: str, batch_size: int, num_workers: int, n_links: int, multi_gpu: bool = False,
        *args, **kwargs
    ):
        self.__data_path = data_path
        self.__scenario = scenario
        self.__n_links = n_links
        super().__init__(batch_size=batch_size, num_workers=num_workers, multi_gpu=multi_gpu)
    
    @staticmethod
    def collate_fn(items):
        bs_location_, ue_location_, toa_, aod_, aoa_, is_los_, path_responses_, image_size_, meta_ = \
            [], [], [], [], [], [], [], [], []
        
        max_len = 0
        for item in items:
            bs_location, ue_location, toa, aod, aoa, is_los, path_responses, image_size, meta = item
            
            l = len(toa)
            if l > max_len:
                max_len = l
        
        for item in items:
            bs_location, ue_location, toa, aod, aoa, is_los, path_responses, image_size, meta = item
            
            bs_location_.append(torch.stack([torch.tensor(bs_location) for _ in range(max_len)]))
            ue_location_.append(torch.tensor(ue_location))
            toa_.append(torch.tensor([(toa[i] if i < len(toa) else 0) for i in range(max_len)]))
            aod_.append(
                torch.stack(
                    [(torch.tensor(aod[i]) if i < len(aod) else torch.zeros_like(torch.tensor(aod[0]))) for i in
                     range(max_len)]
                )
            )
            aoa_.append(
                torch.stack(
                    [(torch.tensor(aoa[i]) if i < len(aoa) else torch.zeros_like(torch.tensor(aoa[0]))) for i in
                     range(max_len)]
                )
            )
            path_responses_.append(
                torch.stack(
                    [(torch.tensor(path_responses[i]) if i < len(path_responses) else torch.zeros_like(
                        torch.tensor(path_responses[0])
                    )) for i in range(max_len)]
                )
            )
            is_los_.append(torch.tensor(is_los))
            image_size_.append(torch.tensor(image_size))
            meta_.append(meta)
        
        bs_location_ = torch.stack(bs_location_)
        ue_location_ = torch.stack(ue_location_)
        toa_ = torch.stack(toa_)
        aod_ = torch.stack(aod_)
        aoa_ = torch.stack(aoa_)
        path_responses_ = torch.stack(path_responses_)
        is_los_ = torch.stack(is_los_)
        image_size_ = torch.stack(image_size_)
        
        return bs_location_, ue_location_, toa_, aod_, aoa_, is_los_, path_responses_, image_size_, meta_
    
    def prepare_data(self) -> None:
        self._train_set = WAIRDDatasetFullAllLinks(
            data_path=self.__data_path, scenario=self.__scenario, split="train",
            n_links=self.__n_links
        )
        self._val_set = WAIRDDatasetFullAllLinks(
            data_path=self.__data_path, scenario=self.__scenario, split="val",
            n_links=self.__n_links
        )
        self._test_set = WAIRDDatasetFullAllLinks(
            data_path=self.__data_path, scenario=self.__scenario, split="test",
            n_links=self.__n_links
        )
