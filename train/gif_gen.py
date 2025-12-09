import os
import sys
import numpy as np
import imageio
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from env.highway_env import TwoLaneHighwayEnv

# -----------------------------------------
# PARAMETERS
# -----------------------------------------
N_SCENARIOS = 10
FPS = 60  # velocidad del gif
SAVE_DIR = "/home/cris/hierarchical_rl_robotaxi/logs/2025-12-08_10-21-13/tensorboard/metrics"
GIF_PATH = os.path.join(SAVE_DIR, "eval_randomized.gif")

os.makedirs(SAVE_DIR, exist_ok=True)
rng = np.random.default_rng(1234)

# -----------------------------------------
# Load environment and model
# -----------------------------------------
env = TwoLaneHighwayEnv(
    config_dir=os.path.join(PROJECT_ROOT, "config"),
    render_mode="human"
)

model = PPO.load(os.path.join(
    PROJECT_ROOT,
    "logs/2025-12-08_10-21-13/models/best_model.zip"
))


# -----------------------------------------
# NEW RANDOMIZER (1 to 7 barrels, s>30m, d constant)
# -----------------------------------------
def randomize_obstacles(env, rng):
    num_barrels = rng.integers(1, 8)  # 1 a 7 barriles

    lane_0_d = env.lane_offsets[0]
    lane_1_d = env.lane_offsets[1]
    lane_options = [lane_0_d, lane_1_d]

    s_min = 30.0
    s_max = env.s_max - 5.0
    s_positions = []
    barrels = []

    for i in range(num_barrels):
        while True:
            candidate_s = rng.uniform(s_min, s_max)
            if all(abs(candidate_s - prev) >= 10.0 for prev in s_positions):
                s_positions.append(candidate_s)
                break

        d_value = lane_options[rng.integers(0, 2)]
        barrels.append({
            "type": "barrel",
            "s": float(candidate_s),
            "d": float(d_value),
            "radius": 0.6
        })

    env.road_obstacles = barrels


# -----------------------------------------
# FRAME COLLECTION
# -----------------------------------------
frames = []

def capture_frame(env):
    """Captures the current matplotlib render frame."""
    env.render()
    fig = env._fig
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    return frame


# -----------------------------------------
# RUN SCENARIOS
# -----------------------------------------
for ep in range(N_SCENARIOS):
    print(f"Running scenario {ep+1}/{N_SCENARIOS}")

    randomize_obstacles(env, rng)  # NEW RANDOM POSITIONS
    obs, info = env.reset()
    done = False

    # Capture initial frame
    frames.append(capture_frame(env))

    step = 0
    while not done and step < 300:  # max steps for display
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        frames.append(capture_frame(env))

        done = terminated or truncated
        step += 1


# -----------------------------------------
# SAVE GIF
# -----------------------------------------
print(f"Saving GIF at {GIF_PATH}")
imageio.mimsave(GIF_PATH, frames, fps=FPS)

print("GIF successfully generated!")
