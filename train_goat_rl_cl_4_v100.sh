#!/bin/bash

#python -m ovon.run \
#  --run-type train \
#  --exp-config config/experiments/transformer_rl_goat.yaml
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
torchrun --standalone --nproc_per_node=4 \
	 -m ovon.run \
	 --run-type train \
	 --exp-config config/experiments/transformer_rl_goat_cl_4v100.yaml \


