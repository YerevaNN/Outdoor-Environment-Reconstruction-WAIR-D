import logging
from typing import Iterable, List

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src.datamodules.wair_d_base import WAIRDBaseDatamodule
from src.utils import EpochCounter, log_hyperparameters

log = logging.getLogger(__name__)


def train(config: DictConfig) -> None:
    epoch_counter = EpochCounter()
    gpus = config.trainer.gpus
    multi_gpu = gpus == -1 or (isinstance(gpus, Iterable) and len(gpus) > 1)
    
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: WAIRDBaseDatamodule = hydra.utils.instantiate(
        config.datamodule, epoch_counter=epoch_counter, multi_gpu=multi_gpu
    )
    
    log.info(f"Instantiating algorithm {config.algorithm._target_}")
    algorithm: LightningModule = hydra.utils.instantiate(
        config.algorithm,
        epoch_counter=epoch_counter,
        network=None,  # instead, we give network_conf
        network_conf=(OmegaConf.to_yaml(config.network) if "network" in config else None),
        optimizer_conf=(OmegaConf.to_yaml(config.optimizer) if "optimizer" in config else None),
        scheduler_conf=(OmegaConf.to_yaml(config.scheduler) if "scheduler" in config else None)
    )
    
    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))
    
    # Init lightning loggers
    loggers: List[LightningLoggerBase] = []
    if "loggers" in config:
        for name, lg_conf in config.loggers.items():
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger = hydra.utils.instantiate(lg_conf)
            loggers.append(logger)
    
    if "strategy" in config:
        log.info(f"Instantiating strategy <{config.strategy}>")
        strategy = hydra.utils.instantiate(config.strategy)
    else:
        if multi_gpu:
            log.error("In case of using multiple GPUs, you must provide a strategy")
        strategy = None
    
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, strategy=strategy, _convert_="partial"
    )
    
    log_hyperparameters(config=config, algorithm=algorithm, trainer=trainer)
    
    # Train the model
    log.info("Starting training!")
    trainer.fit(algorithm, datamodule=datamodule)
    
    trainer.test(dataloaders=datamodule.test_dataloader())
