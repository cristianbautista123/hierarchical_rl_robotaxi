#!/bin/bash
#SBATCH -J hier_ppo
#SBATCH -A PAS2099
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err


# Create timestamp folder
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_DIR=$HOME/hierarchical_rl_robotaxi/logs/$TIMESTAMP
mkdir -p $RUN_DIR

# Redirect stdout and stderr to our RUN_DIR
exec > >(tee -a "$RUN_DIR/train.out") 2> >(tee -a "$RUN_DIR/train.err" >&2)

echo "Run directory: $RUN_DIR"
cd $HOME/hierarchical_rl_robotaxi


python visualization/plot_map.py --output_dir $RUN_DIR

python train/train_ppo.py --run_dir $RUN_DIR
