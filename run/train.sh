#!/bin/sh

HYDRA_FULL_ERROR=1 \
python run.py \
loggers=[] \
datamodule.batch_size=40 \
trainer.gpus=[0]
