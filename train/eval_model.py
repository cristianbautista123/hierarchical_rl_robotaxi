import os
import sys
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from env.highway_env import TwoLaneHighwayEnv

env = TwoLaneHighwayEnv(
    config_dir=os.path.join(os.path.dirname(__file__), "..", "config"),
    render_mode="human",
)

model_path = os.path.join(PROJECT_ROOT, "logs", "2025-12-08_10-21-13", "models", "best_model.zip")
model = PPO.load(model_path)

obs, info = env.reset()

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
