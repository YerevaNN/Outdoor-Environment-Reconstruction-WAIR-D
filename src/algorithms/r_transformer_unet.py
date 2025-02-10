import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.algorithms.r_unet import RUNet

log = logging.getLogger(__name__)


class RTransformerUNet(RUNet):
    
    def __init__(
        self,
        use_ce: bool,
        use_dice: bool,
        optimizer_conf: DictConfig = None,
        scheduler_conf: DictConfig = None,
        network: nn.Module = None,
        network_conf: DictConfig = None,
        gpu: int = None,
        *args, **kwargs
    ):
        super().__init__(
            use_ce=use_ce, use_dice=use_dice,
            optimizer_conf=optimizer_conf, scheduler_conf=scheduler_conf,
            network=network, network_conf=network_conf, gpu=gpu
        )
    
    def pred(self, batch):
        input_image, supervision_image, sequences, _ = batch
        pred_image = self._network(
            torch.Tensor([input_image]).cuda(self._gpu), torch.Tensor([sequences]).cuda(self._gpu)
        )
        return pred_image
    
    def _step(self, batch, *args, **kwargs):
        input_image, supervision_image, sequences, _ = batch
        pred_image = self._network(input_image, sequences)
        return self.get_metrics(pred_image, supervision_image)
