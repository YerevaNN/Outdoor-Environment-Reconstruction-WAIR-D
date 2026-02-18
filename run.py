import logging
import os
import random
import warnings

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

log = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

hydra.core.global_hydra.GlobalHydra.instance().clear()


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(config: DictConfig) -> None:
    from src import (
        utils, train, prepare_data, pred,
        evaluate,
    )
    
    if config.seed == -1:
        config.seed = random.randint(0, 10 ** 8)
    
    seed_everything(config.seed)
    
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    
    if config.get("print_config"):
        utils.print_config(config, fields=tuple(config.keys()), resolve=True)
    
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")
    
    if config.name == "prepare_data":
        return prepare_data(config)
    
    if config.name == "train":
        return train(config)
    
    if config.name == "inference":
        return pred(config)
    
    if config.name == "evaluation":
        return evaluate(config)


if __name__ == "__main__":
    main()
