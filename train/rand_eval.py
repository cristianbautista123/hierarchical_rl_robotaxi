import os
import sys
import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from env.highway_env import TwoLaneHighwayEnv

N_EPISODES = 100
rng = np.random.default_rng(123)

env = TwoLaneHighwayEnv(
    config_dir=os.path.join(PROJECT_ROOT, "config"),
    render_mode=None,   # TURN OFF FOR SPEED
)

model = PPO.load(os.path.join(PROJECT_ROOT,
                              "logs/2025-12-08_10-21-13/models/best_model.zip"))

results = {
    "success": 0,
    "collisions": 0,
    "episode_rewards": [],
    "episode_lengths": [],
    "lane_changes": [],
    "min_distances": [],
}

# === Helper: detect lane-changes
def count_lane_changes(lane_history):
    return sum(1 for i in range(1, len(lane_history)) if lane_history[i] != lane_history[i-1])


for ep in range(N_EPISODES):

    # Randomize obstacle positions BEFORE each episode
    env.randomize_obstacles(rng)

    obs, info = env.reset()

    done = False
    ep_reward = 0
    ep_steps = 0
    lane_history = []
    min_dist = 999

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        ep_reward += reward
        ep_steps += 1
        lane_history.append(info["lane"])

        # Update minimum distance to obstacles
        dist_curr, dist_other = obs[3], obs[4]
        min_dist = min(min_dist, dist_curr, dist_other)

        if terminated:
            done = True
            # Check collision
            if reward < -40:  # collision penalty
                results["collisions"] += 1
            else:
                results["success"] += 1

        if truncated:
            done = True

    # Save metrics
    results["episode_rewards"].append(ep_reward)
    results["episode_lengths"].append(ep_steps)
    results["lane_changes"].append(count_lane_changes(lane_history))
    results["min_distances"].append(min_dist)

# Final Stats
print("\n=== Evaluation Results ===")
print(f"Success rate: {results['success']}/{N_EPISODES}")
print(f"Collision rate: {results['collisions']}/{N_EPISODES}")
print(f"Avg reward: {np.mean(results['episode_rewards']):.2f}")
print(f"Avg episode length: {np.mean(results['episode_lengths']):.1f}")
print(f"Avg lane changes: {np.mean(results['lane_changes']):.2f}")
print(f"Avg minimum distance to obstacle: {np.mean(results['min_distances']):.2f}")
