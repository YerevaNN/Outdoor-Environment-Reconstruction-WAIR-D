import logging

import hydra
from omegaconf import DictConfig
from tabulate import tabulate

from src.datamodules.wair_d_base import WAIRDBaseDatamodule
from src.evaluators import ReconstructionEvaluation
from src.utils import EpochCounter

log = logging.getLogger(__name__)


def evaluate(config: DictConfig) -> None:
    epoch_counter = EpochCounter()
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: WAIRDBaseDatamodule = hydra.utils.instantiate(config.datamodule, epoch_counter=epoch_counter)
    datamodule.prepare_data()
    
    evaluator = ReconstructionEvaluation(config.prediction_path, datamodule.test_set)
    metrics = evaluator.get_all_metrics()
    log.info(("\n" + tabulate([metrics], headers=["IoU", "IoG", "IoP", "Hausdorff", "Chamfer"])))
