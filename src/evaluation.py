import logging

import hydra
from omegaconf import DictConfig
from tabulate import tabulate

from src.datamodules.wair_d_base import WAIRDBaseDatamodule
from src.evaluators import LocalizationEvaluation, ReconstructionEvaluation
from src.utils import EpochCounter

log = logging.getLogger(__name__)


def evaluate(config: DictConfig) -> None:
    epoch_counter = EpochCounter()
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: WAIRDBaseDatamodule = hydra.utils.instantiate(config.datamodule, epoch_counter=epoch_counter)
    datamodule.prepare_data()
    
    if config.task == "localization":
        evaluator = LocalizationEvaluation(config.prediction_path, datamodule.test_set)
        rmses = evaluator.get_rmse_all_los_nlos()
        accuracies = evaluator.get_accuracy_all_los_nlos(config.allowable_errors)
        log.info(("\n" + tabulate(zip(["All", "LOS", "NLOS"], rmses), headers=["RMSE"])))
        log.info(
            "\n" + tabulate(
                [
                    ["All"] + accuracies[0],
                    ["LOS"] + accuracies[1],
                    ["NLOS"] + accuracies[2]
                ],
                headers=[f"{t}m acc" for t in config.allowable_errors]
            )
        )
    elif config.task == "reconstruction":
        evaluator = ReconstructionEvaluation(config.prediction_path, datamodule.test_set)
        metrics = evaluator.get_all_metrics()
        log.info(("\n" + tabulate([metrics], headers=["IoU", "IoG", "IoP", "Hausdorff", "Chamfer"])))
    else:
        raise ValueError(f"Unknown task <{config.task}")
