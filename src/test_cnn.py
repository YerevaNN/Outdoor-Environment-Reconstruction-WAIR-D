import logging
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src.datamodules.wair_d_base import WAIRDBaseDatamodule
from src.utils import log_hyperparameters

log = logging.getLogger(__name__)


def test_cnn(config: DictConfig) -> None:
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: WAIRDBaseDatamodule = hydra.utils.instantiate(config.datamodule)
    datamodule.prepare_data()
    
    log.info(f"Instantiating algorithm {config.algorithm._target_} with checkpoint {config.checkpoint_path}")
    
    algorithm: LightningModule = hydra.utils.get_class(config.algorithm._target_) \
        .load_from_checkpoint(
        config.checkpoint_path, **config.algorithm,
        network_conf=(OmegaConf.to_yaml(
            config.network
        ) if "network" in config else None),
        optimizer_conf=(OmegaConf.to_yaml(
            config.optimizer
        ) if "optimizer" in config else None),
        scheduler_conf=(OmegaConf.to_yaml(
            config.scheduler
        ) if "scheduler" in config else None)
        )
    # Init lightning loggers
    loggers: List[LightningLoggerBase] = []
    if "loggers" in config:
        for name, lg_conf in config.loggers.items():
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger = hydra.utils.instantiate(lg_conf)
            loggers.append(logger)
    
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, logger=loggers, _convert_="partial"
    )
    
    log_hyperparameters(config=config, algorithm=algorithm, trainer=trainer)
    
    trainer.test(model=algorithm, dataloaders=datamodule.test_dataloader())
