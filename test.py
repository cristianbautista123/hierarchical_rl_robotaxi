from env import TwoLaneHighwayEnv

env = TwoLaneHighwayEnv(render_mode="human")

obs, info = env.reset()
print("obs0:", obs, "info0:", info)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward, info)
    if terminated or truncated:
        break

env.close()
