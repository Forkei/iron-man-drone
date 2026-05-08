#!/bin/bash
export PYTHONPATH=/mnt/c/Users/forke/Documents/Drone/iron-man-drone/src
cd /mnt/c/Users/forke/Documents/Drone/iron-man-drone
exec /home/forke/jax_env/bin/python -u scripts/train_m2.py \
  --config /mnt/c/Users/forke/Documents/Drone/iron-man-drone/experiments/m2_no_dr_ablation/config.yaml \
  --num_envs 2048 \
  >/mnt/c/Users/forke/Documents/Drone/iron-man-drone/m2_no_dr_ablation.log 2>&1
