import logging
import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torch import nn
from tqdm import tqdm

from src.datamodules.wair_d_base import WAIRDBaseDatamodule
from src.utils import EpochCounter

log = logging.getLogger(__name__)


def pred(config: DictConfig) -> None:
    epoch_counter = EpochCounter()
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: WAIRDBaseDatamodule = hydra.utils.instantiate(config.datamodule, epoch_counter=epoch_counter)
    datamodule.prepare_data()
    
    log.info(f"Instantiating algorithm {config.algorithm._target_} with checkpoint {config.checkpoint_path}")
    algorithm: LightningModule = hydra.utils.get_class(config.algorithm._target_).load_from_checkpoint(
        config.checkpoint_path, **config.algorithm,
        network_conf=(OmegaConf.to_yaml(config.network) if "network" in config else None),
        gpu=config.gpu,
        epoch_counter=epoch_counter
    )
    algorithm.network.eval()
    algorithm.network.cuda(config.gpu)
    
    pred_path = os.path.join(
        config.prediction_dir,
        os.path.basename(os.path.dirname(os.path.dirname(config.checkpoint_path)))
    )
    os.makedirs(pred_path, exist_ok=True)
    
    if config.split == "test":
        for i, batch in tqdm(enumerate(datamodule.test_set), total=len(datamodule.test_set)):
            out = nn.functional.sigmoid(algorithm.pred(batch)).detach().cpu().numpy()[0, 0]
            np.savez(os.path.join(pred_path, str(i)), out)
    elif config.split == "scenario2":
        for env in tqdm(list(os.listdir(config.datamodule.scenario2_path))):
            idx = datamodule.val_set._environments.index(env)
            batch = datamodule.val_set[idx]
            out = nn.functional.sigmoid(algorithm.pred(batch)).detach().cpu().numpy()[0, 0]
            np.savez(os.path.join(pred_path, env), out)
    else:
        raise ValueError(f"Unknown split {config.split}")
