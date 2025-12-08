import os
import sys
import json
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from env.highway_env import TwoLaneHighwayEnv


# ---------------------------------------------------------
# Parse run directory from command line
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True,
                    help="Folder where logs/models/checkpoints will be saved")
args = parser.parse_args()

RUN_DIR = args.run_dir

# Create structured subfolders
os.makedirs(os.path.join(RUN_DIR, "tensorboard"), exist_ok=True)
os.makedirs(os.path.join(RUN_DIR, "eval_logs"), exist_ok=True)
os.makedirs(os.path.join(RUN_DIR, "ppo_checkpoints"), exist_ok=True)
os.makedirs(os.path.join(RUN_DIR, "models"), exist_ok=True)

# ---------------------------------------------------------
# Environment factory
# ---------------------------------------------------------
def make_env():
    env = TwoLaneHighwayEnv(
        config_dir=os.path.join(PROJECT_ROOT, "config"),
        render_mode=None,
        lane_change_penalty=0.0,
    )
    env = Monitor(env)
    return env

env = make_env()

# ---------------------------------------------------------
# PPO Model
# ---------------------------------------------------------
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log=os.path.join(RUN_DIR, "tensorboard"),
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
)

# ---------------------------------------------------------
# Checkpoint callback
# ---------------------------------------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=os.path.join(RUN_DIR, "ppo_checkpoints"),
    name_prefix="ppo_highway",
)

# ---------------------------------------------------------
# Evaluation callback
# ---------------------------------------------------------
eval_callback = EvalCallback(
    make_env(),
    best_model_save_path=os.path.join(RUN_DIR, "models"),
    log_path=os.path.join(RUN_DIR, "eval_logs"),
    eval_freq=10_000,
    deterministic=True,
    render=False,
)

# ---------------------------------------------------------
# Train PPO
# ---------------------------------------------------------
model.learn(
    total_timesteps=300_000,
    callback=[checkpoint_callback, eval_callback]
)

# ---------------------------------------------------------
# Save hyperparameters (clean and JSON-safe)
# ---------------------------------------------------------
config = {
    "policy": model.policy_class.__name__,
    "algorithm": "PPO",
    "gamma": model.gamma,
    "learning_rate": 3e-4,
    "n_steps": model.n_steps,
    "batch_size": model.batch_size,
    "ent_coef": model.ent_coef,
    "clip_range": 0.2,
    "total_timesteps": 300_000
}

with open(os.path.join(RUN_DIR, "hparams.json"), "w") as f:
    json.dump(config, f, indent=4)

# ---------------------------------------------------------
# Save final model
# ---------------------------------------------------------
model.save(os.path.join(RUN_DIR, "models", "ppo_highway_final"))

print("\nTraining complete!")
print(f"Run directory: {RUN_DIR}\n")
