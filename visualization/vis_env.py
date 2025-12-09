import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from env.highway_env import TwoLaneHighwayEnv

# Create environment in RENDER MODE
env = TwoLaneHighwayEnv(
    config_dir=os.path.join(PROJECT_ROOT, "config"),
    render_mode="human",      # <<--- REQUIRED
)

obs, info = env.reset()

# Run a simple loop to visualize
for step in range(200):
    action = 0   # stay in lane (just for visualization)
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.1)  # slows down plotting
    if terminated:
        break

env.close()
