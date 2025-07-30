#!/bin/bash


python -m ovon.run   --run-type eval   --exp-config config/experiments/transformer_dagger.yaml   habitat_baselines.eval_ckpt_path_dir=/home/fsq/ovon/data/eval_checkpoints/ckpt.12.pth
