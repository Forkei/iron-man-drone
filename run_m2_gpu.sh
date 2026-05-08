#!/bin/bash
export PYTHONPATH=/mnt/c/Users/forke/Documents/Drone/iron-man-drone/src
cd /mnt/c/Users/forke/Documents/Drone/iron-man-drone
exec /home/forke/jax_env/bin/python -u scripts/train_m2.py \
  --resume_from /mnt/c/Users/forke/Documents/Drone/iron-man-drone/experiments/m2_phase1_baseline/m2_phase1_baseline_1778239815/checkpoints/epoch_013000 \
  --start_epoch 13000 \
  --resume_med_nominal 0.097 \
  --trend_gate_improvement 0.005 \
  --total_epochs 15000 \
  --num_envs 2048 \
  >/mnt/c/Users/forke/Documents/Drone/iron-man-drone/m2_resume.log 2>&1
