# !/bin/bash

# This script needs 2 gpus
CUDA_VISIBLE_DEVICES=0,1 python3 svd_reinforce_hydra.py
