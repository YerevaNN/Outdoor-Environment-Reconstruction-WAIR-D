import torch
from omegaconf import DictConfig
from torch import nn

from src.algorithms.erm import ERM


class TransformerERM(ERM):
    
    def __init__(
        self,
        groups_count: int,
        allowable_errors: list[int],
        optimizer_conf: DictConfig = None,
        scheduler_conf: DictConfig = None,
        network: nn.Module = None,
        network_conf: DictConfig = None,
        gpu: int = None,
        *args, **kwargs
    ):
        super().__init__(
            groups_count, allowable_errors,
            optimizer_conf, scheduler_conf,
            network, network_conf, gpu,
            *args, **kwargs
        )
    
    def pred(self, batch):
        sequence, mask, ue_location, image_size, is_los = batch
        ue_location_pred = self._network(torch.Tensor([sequence]).cuda(self._gpu), torch.Tensor([mask]).cuda(self._gpu))
        return ue_location_pred
    
    def _step(self, batch, *args, **kwargs):
        sequence, mask, ue_location, image_size, is_los = batch
        ue_location_pred = self._network(sequence, mask)
        return self.get_metrics(ue_location, image_size, is_los, ue_location_pred)
