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

model = PPO.load("ppo_highway_final")

obs, info = env.reset()

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
