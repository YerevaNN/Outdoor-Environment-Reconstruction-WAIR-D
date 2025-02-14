import logging
from typing import Any, List

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.utils import dice_loss, EpochCounter

log = logging.getLogger(__name__)


class UNetERM(pl.LightningModule):
    
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
        super().__init__()
        
        self.__optimizer_conf = optimizer_conf
        self.__scheduler_conf = scheduler_conf
        self._groups_count = groups_count  # TODO do we really need this or we can mine this info from the data
        self._allowable_errors = allowable_errors
        self._use_dice = use_dice
        self.__epoch_counter = epoch_counter
        
        if network is None:
            self._network = hydra.utils.instantiate(OmegaConf.create(network_conf))
        else:
            self._network = network
        
        self._gpu = gpu
        if self._gpu is not None:
            self.network.cuda(gpu)
        
        self._mse = nn.MSELoss(reduction='none')
        self._group_weights = self.__get_initial_group_weights()  # sets (1/n, 1/n, 1/n, ..., 1/n)
    
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
        input_image, supervision_image, ue_location, image_size, is_los = batch
        pred_image = self._network(torch.Tensor([input_image]).cuda(self._gpu))
        return pred_image
    
    def _step(self, batch, *args, **kwargs):
        input_image, supervision_image, ue_location, image_size, is_los = batch
        pred_image = self._network(input_image)
        return self.get_metrics(supervision_image, ue_location, image_size, is_los, pred_image)
    
    def get_metrics(
        self,
        supervision_image, ue_location, image_size, is_los,
        pred_image
    ):
        max_ind = pred_image.flatten(1).argmax(dim=-1)
        ue_location_pred = torch.stack(
            [max_ind % max(pred_image[0][0].shape), max_ind // max(pred_image[0][0].shape)], dim=1
        )  # .cuda()
        
        mses_meters = self._mse(
            ue_location_pred, ue_location
        ).sum(dim=1).sqrt() * image_size / max(pred_image[0][0].shape)
        
        accuracies = {f"acc_{p}": (mses_meters < p).sum() / len(mses_meters) for p in self._allowable_errors}
        
        mse_meters = mses_meters.mean()
        
        group_metrics = self._get_group_metrics(is_los, mses_meters, self._allowable_errors)
        
        loss = nn.functional.binary_cross_entropy_with_logits(pred_image, supervision_image)
        if self._use_dice:
            loss += dice_loss(nn.functional.sigmoid(pred_image)[:, 0], supervision_image[:, 0], multiclass=False)
        
        metrics = {
            'loss': loss,
            **{acc: acc_val.to('cpu').detach() for acc, acc_val in accuracies.items()},
            'mse_meters': mse_meters.to('cpu').detach(),
            "group_metrics": group_metrics
        }
        
        return metrics
    
    def training_step(self, batch, *args, **kwargs):
        return self._step(batch)
    
    def validation_step(self, batch, *args, **kwargs):
        return self._step(batch)
    
    def test_step(self, batch, *args, **kwargs):
        return self._step(batch)
    
    def on_train_epoch_start(self):
        self.__epoch_counter.count = self.current_epoch
    
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
    
    def __get_initial_group_weights(self):
        group_weights = torch.ones(self._groups_count)
        group_weights = group_weights / group_weights.sum()
        group_weights = group_weights.to('cpu')
        
        return group_weights
    
    def _get_group_metrics(self, z, mses_meters, allowable_errors: list[int]):
        group_metrics = {}
        for group_i in range(self._groups_count):
            group_mses_meters = mses_meters[z == group_i]
            group_count = len(group_mses_meters)
            
            group_mse_meters = 0 if group_count == 0 else group_mses_meters.mean().to('cpu').detach()
            
            group_accuracies = {
                f"acc_{p}": 0 if group_count == 0 else (group_mses_meters < p).sum().to('cpu').detach() / group_count
                for p in allowable_errors
            }
            
            group_metrics = {
                **group_metrics,
                group_i: {
                    "metrics": {
                        f"mse_meters": group_mse_meters,
                        **group_accuracies
                    },
                    "count": len(group_mses_meters)
                }
            }
        
        return group_metrics
    
    def __calculate_epoch_metrics(self, outputs: List[Any]) -> dict:
        general_metric_names = [k for k in outputs[0].keys() if k != "group_metrics"]
        group_metric_names = [k for k in outputs[0]["group_metrics"][0]["metrics"].keys()]
        
        # init combined metrics with zero values
        combined_general_metrics = {k: 0 for k in general_metric_names}
        
        combined_group_metrics = {
            i: dict(
                metrics=dict(**{metric: 0 for metric in group_metric_names}),
                count=0
            ) for i in range(self._groups_count)
        }
        
        # add all output values to combined_group_metrics
        for o in outputs:
            for group_i, group_metrics_obj in o['group_metrics'].items():
                for metric_name, metric_val in group_metrics_obj["metrics"].items():
                    combined_group_metrics[group_i]['metrics'][metric_name] += metric_val * group_metrics_obj["count"]
                combined_group_metrics[group_i]["count"] += group_metrics_obj["count"]
            
            for k in o.keys():
                if k != "group_metrics":
                    combined_general_metrics[k] += o[k]
        
        # compute means of metrics
        for group_i, group_metrics_obj in combined_group_metrics.items():
            for metric_name, metric_val in group_metrics_obj["metrics"].items():
                combined_group_metrics[group_i]['metrics'][metric_name] /= combined_group_metrics[group_i][
                                                                               "count"] + 1e-6
        
        for k in outputs[0].keys():
            if k != "group_metrics":
                combined_general_metrics[k] /= len(outputs)
        
        # worst case info calculation
        worst_case_info = {k: [] for k in combined_group_metrics[0]["metrics"].keys()}
        for group_metrics_obj in combined_group_metrics.values():
            for metric_name, metric_val in group_metrics_obj["metrics"].items():
                worst_case_info[metric_name].append(metric_val)
        
        worst_case_info = {f"worst_group_{k}": (min(vals) if "acc" in k else max(vals))
                           for k, vals in worst_case_info.items()}
        
        # merge group metrics into one dict
        combined_group_metrics = {f"group_{i}_{metric_name}": metric_val for i, m in combined_group_metrics.items()
                                  for metric_name, metric_val in {**m["metrics"], "count": m["count"]}.items()}
        
        # merge all
        epoch_metrics_sep = {
            **combined_general_metrics,
            **combined_group_metrics,
            **worst_case_info
        }
        
        epoch_metrics_shared = {
            "learning_rate": self.trainer.optimizers[0].param_groups[0]["lr"]
        }
        for i in range(self._groups_count):
            epoch_metrics_sep[f"w_group_{i}"] = self._group_weights[i]
        
        if self.logger:
            self.logger.log_metrics(epoch_metrics_shared, self.trainer.current_epoch)
        else:
            log.info(f"""\n{epoch_metrics_shared}\n""")
        
        return epoch_metrics_sep
