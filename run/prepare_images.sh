#!/bin/sh

HYDRA_FULL_ERROR=1 \
python ../run.py \
--config-name=prepare_data \
prepared_data_dir=/nfs/dgx/raid/iot/data/rec_revisit/wair_d_r_images/scenario_1
