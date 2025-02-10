import torch
from omegaconf import DictConfig
from torch import nn

from src.algorithms.unet_erm import UNetERM
from src.utils import EpochCounter


class MLPUNet(UNetERM):
    
    def __init__(
        self,
        groups_count: int,
        allowable_errors: list[int],
        use_dice: bool,
        epoch_counter: EpochCounter,
        optimizer_conf: DictConfig = None,
        scheduler_conf: DictConfig = None,
        network: nn.Module = None,
        network_conf: DictConfig = None,
        gpu: int = None,
        *args, **kwargs
    ):
        super().__init__(
            groups_count, allowable_errors, use_dice,
            epoch_counter, optimizer_conf, scheduler_conf, network,
            network_conf, gpu,
            *args, **kwargs
        )
    
    def pred(self, batch):
        input_image, sequence, _, supervision_image, ue_location, image_size, is_los = batch
        pred_image = self._network(
            torch.Tensor([input_image]).cuda(self._gpu), torch.Tensor([sequence]).cuda(self._gpu)
        )
        return pred_image
    
    def _step(self, batch, *args, **kwargs):
        input_image, sequence, _, supervision_image, ue_location, image_size, is_los = batch
        pred_image = self._network(input_image, sequence)
        return self.get_metrics(supervision_image, ue_location, image_size, is_los, pred_image)
