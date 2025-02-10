import logging
from typing import Any, List

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.utils import dice_loss, iou_score

log = logging.getLogger(__name__)


class RUNet(pl.LightningModule):
    
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
        super().__init__()
        
        self.__optimizer_conf = optimizer_conf
        self.__scheduler_conf = scheduler_conf
        
        assert use_ce or use_dice, "Loss function is not specified."
        self._use_ce = use_ce
        self._use_dice = use_dice
        
        if network is None:
            self._network = hydra.utils.instantiate(OmegaConf.create(network_conf))
        else:
            self._network = network
        
        self._gpu = gpu
        if self._gpu is not None:
            self.network.cuda(gpu)
    
    @property
    def network(self):
        return self._network
    
    def forward(self, *args, **kwargs):
        outputs = self._network(*args, **kwargs)
        return outputs
    
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            OmegaConf.create(self.__optimizer_conf),
            params=filter(lambda p: p.requires_grad, self.parameters())
        )
        
        ret_opt = {"optimizer": optimizer}
        if self.__scheduler_conf is not None:
            scheduler_conf = OmegaConf.create(self.__scheduler_conf)
            monitor = scheduler_conf.monitor
            del scheduler_conf.monitor
            
            scheduler = hydra.utils.instantiate(scheduler_conf, optimizer=optimizer)
            sch_opt = {"scheduler": scheduler, "monitor": monitor}
            
            ret_opt.update({"lr_scheduler": sch_opt})
        
        return ret_opt
    
    def pred(self, batch):
        input_image, supervision_image = batch
        pred_image = self._network(torch.Tensor([input_image]).cuda(self._gpu))
        return pred_image
    
    def _step(self, batch, *args, **kwargs):
        input_image, supervision_image = batch
        pred_image = self._network(input_image)
        return self.get_metrics(pred_image, supervision_image)
    
    def get_metrics(self, pred_image, supervision_image):
        pred_image_sigmoid = torch.sigmoid(pred_image)
        
        loss = 0
        if self._use_ce:
            loss += nn.functional.binary_cross_entropy_with_logits(pred_image, supervision_image)
        if self._use_dice:
            loss += dice_loss(pred_image_sigmoid[:, 0], supervision_image[:, 0], multiclass=False)
        
        iou = iou_score(pred_image_sigmoid, supervision_image)
        
        metrics = {
            "loss": loss,
            "iou": iou
        }
        
        return metrics
    
    def training_step(self, batch, *args, **kwargs):
        return self._step(batch)
    
    def validation_step(self, batch, *args, **kwargs):
        return self._step(batch)
    
    def test_step(self, batch, *args, **kwargs):
        return self._step(batch)
    
    def __epoch_end(self, outputs: List[Any], split_name):
        epoch_metrics = self.__calculate_epoch_metrics(outputs)
        epoch_metrics = {f'{split_name}_{key}': epoch_metrics[key] for key in epoch_metrics}
        
        self.trainer.callback_metrics.update(epoch_metrics)
        if self.logger:
            self.logger.log_metrics(epoch_metrics, self.trainer.current_epoch)
        else:
            log.info(f"""\n{epoch_metrics}\n""")
    
    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.__epoch_end(outputs, split_name='train')
    
    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.__epoch_end(outputs, split_name='val')
    
    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.__epoch_end(outputs, split_name='test')
    
    def __calculate_epoch_metrics(self, outputs: List[Any]) -> dict:
        # init combined metrics with zero values
        combined_general_metrics = {k: 0 for k in outputs[0].keys()}
        
        # add all output values to combined_group_metrics
        for o in outputs:
            for k in o.keys():
                combined_general_metrics[k] += o[k]
        
        # compute means of metrics
        for k in outputs[0].keys():
            combined_general_metrics[k] /= len(outputs)
        
        # merge all
        epoch_metrics_sep = combined_general_metrics
        
        epoch_metrics_shared = {
            "learning_rate": self.trainer.optimizers[0].param_groups[0]["lr"]
        }
        
        if self.logger:
            self.logger.log_metrics(epoch_metrics_shared, self.trainer.current_epoch)
        else:
            log.info(f"""\n{epoch_metrics_shared}\n""")
        
        return epoch_metrics_sep
