#!/bin/bash

# ===== Create timestamp folder =====
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_DIR=$HOME/hierarchical_rl_robotaxi/logs/$TIMESTAMP
mkdir -p $RUN_DIR

# ===== Redirect stdout and stderr to log files =====
exec > >(tee -a "$RUN_DIR/train.out") 2> >(tee -a "$RUN_DIR/train.err" >&2)

echo "Run directory: $RUN_DIR"
cd $HOME/hierarchical_rl_robotaxi

# ===== Run map visualization =====
python visualization/plot_map.py --output_dir $RUN_DIR

# ===== Run PPO training =====
python train/train_ppo.py --run_dir $RUN_DIR
