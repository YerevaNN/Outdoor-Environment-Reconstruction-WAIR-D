#!/bin/sh

HYDRA_FULL_ERROR=1 \
python run.py \
--config-name=prepare_data \
sequence=True \
reconstruction=False \
prepared_data_dir=/path/to/sequences
