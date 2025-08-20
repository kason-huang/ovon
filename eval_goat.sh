#!/bin/bash


python -m ovon.run   --run-type eval   --exp-config config/experiments/transformer_rl_goat_eval.yaml   habitat_baselines.eval_ckpt_path_dir=/root/workspace/lab/ovon/data/new_checkpoints_goat_0815/latest.pth
