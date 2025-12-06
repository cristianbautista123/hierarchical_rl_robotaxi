import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from env.highway_env import TwoLaneHighwayEnv

# ----------------------------
# Create environment instance
# ----------------------------
env = TwoLaneHighwayEnv(
    config_dir=os.path.join(os.path.dirname(__file__), "..", "config"),
    render_mode=None,          # no render while training
    lane_change_penalty=0.5,   # slightly penalize lane changes
)

# ----------------------------
# PPO Model
# ----------------------------
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
)

# ----------------------------
# Save checkpoints every N steps
# ----------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./ppo_checkpoints/",
    name_prefix="ppo_highway",
)

# ----------------------------
# Train
# ----------------------------
model.learn(
    total_timesteps=300_000,
    callback=checkpoint_callback
)

# Save final model
model.save("ppo_highway_final")
print("\nTraining complete!")
