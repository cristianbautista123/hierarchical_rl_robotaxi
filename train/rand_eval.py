import os
import sys
import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from env.highway_env import TwoLaneHighwayEnv

# ===========================
# Número de evaluaciones
# ===========================
N_EPISODES = 500
rng = np.random.default_rng(42)

# ===========================
# Crear entorno y cargar modelo
# ===========================
env = TwoLaneHighwayEnv(
    config_dir=os.path.join(PROJECT_ROOT, "config"),
    render_mode=None  # desactivar render para velocidad
)

model = PPO.load(os.path.join(
    PROJECT_ROOT, "logs/2025-12-08_10-21-13/models/best_model.zip"
))


# ===========================
# Randomizador externo
# ===========================
def randomize_obstacles(env, rng):
    """
    Randomiza entre 1 y 7 barriles.
    - s > 30 m
    - mínima distancia entre barriles = 10 m
    - d se mantiene fija dependiendo del lane original del obstáculo
    """

    # Número aleatorio de barriles entre 1 y 7
    num_barrels = rng.integers(15, 25)

    # Extraer lanes válidos desde el entorno
    lane_0_d = env.lane_offsets[0]           # normalmente 0.0
    lane_1_d = env.lane_offsets[1]           # ~3.5 m
    lane_options = [lane_0_d, lane_1_d]

    # Rango permitido para s
    s_min = 30.0
    s_max = env.s_max - 5.0   # para evitar que quede al final del mapa

    barrels = []
    s_positions = []

    for i in range(num_barrels):

        # Generar s nuevo cumpliendo distancia mínima de 10 m
        while True:
            candidate_s = rng.uniform(s_min, s_max)

            if all(abs(candidate_s - prev_s) >= 10.0 for prev_s in s_positions):
                s_positions.append(candidate_s)
                break

        # Elegir un lane aleatorio (pero no modificar d)
        d_value = lane_options[rng.integers(0, 2)]

        barrels.append({
            "type": "barrel",
            "s": float(candidate_s),
            "d": float(d_value),
            "radius": 0.6
        })

    env.road_obstacles = barrels



# ===========================
# Métricas
# ===========================
results = {
    "success": 0,
    "collisions": 0,
    "episode_rewards": [],
    "episode_lengths": [],
    "min_dist": [],
    "lane_changes": []
}

def count_lane_changes(lane_history):
    return sum(1 for i in range(1, len(lane_history)) if lane_history[i] != lane_history[i-1])


# ===========================
# Loop de evaluación
# ===========================
for ep in range(N_EPISODES):

    # randomizar ANTES de reset
    randomize_obstacles(env, rng)

    obs, info = env.reset()
    done = False

    ep_reward = 0
    ep_steps = 0
    min_distance = 999
    lane_history = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        ep_reward += reward
        ep_steps += 1
        
        lane_history.append(info["lane"])

        # distancia mínima observada
        min_distance = min(min_distance, float(obs[3]), float(obs[4]))

        if terminated:
            # identificar colisión por reward grande negativo
            if reward <= -40:
                results["collisions"] += 1
            else:
                results["success"] += 1
            done = True

        if truncated:
            done = True

    # guardar métricas episodicas
    results["episode_rewards"].append(ep_reward)
    results["episode_lengths"].append(ep_steps)
    results["min_dist"].append(min_distance)
    results["lane_changes"].append(count_lane_changes(lane_history))


# ===========================
# Resultados finales
# ===========================
print("\n===== RESULTS =====")
print(f"Success rate:   {results['success']}/{N_EPISODES}")
print(f"Collision rate: {results['collisions']}/{N_EPISODES}")
print(f"Avg Reward:     {np.mean(results['episode_rewards']):.2f}")
print(f"Avg Length:     {np.mean(results['episode_lengths']):.1f}")
print(f"Avg Min Dist:   {np.mean(results['min_dist']):.2f}")
print(f"Avg Lane Chg:   {np.mean(results['lane_changes']):.2f}")
