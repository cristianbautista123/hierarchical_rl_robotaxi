import os
import sys
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.frenet import FrenetFrame


class TwoLaneHighwayEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        config_dir=None,
        dt=0.1,
        v_ref=10.0,
        d_step_max=0.4,
        lane_change_penalty=0.0,
        centerline_main_file="main_centerline_fixed.npy",
        centerline_other_file="other_centerline_fixed.npy",
        render_mode=None,
    ):
        super().__init__()

        # ===========================
        # Paths
        # ===========================
        if config_dir is None:
            config_dir = os.path.join(PROJECT_ROOT, "config")

        self.config_dir = config_dir
        yaml_path = os.path.join(config_dir, "obstacles.yaml")

        # ===========================
        # Load map
        # ===========================
        self.cl_main = np.load(os.path.join(config_dir, centerline_main_file))
        self.cl_other = np.load(os.path.join(config_dir, centerline_other_file))

        self.frenet_main = FrenetFrame.from_centerline(self.cl_main)
        self.s_max = float(self.frenet_main.s_accum[-1])

        # Compute lane offsets automatically
        d_list = []
        for p in self.cl_other:
            _, d_i = self.frenet_main.xy_to_sd(p)
            d_list.append(d_i)
        self.lane_offsets = np.array([0.0, float(np.mean(d_list))])

        # ===========================
        # Load ALL YAML obstacles
        # ===========================
        (
            self.road_obstacles,
            self.roadside_objects,
            self.construction_clusters,
            self.scenario_randomization
        ) = self._load_yaml_obstacles(yaml_path)

        # ===========================
        # Environment dynamics params
        # ===========================
        self.dt = dt
        self.v_ref = v_ref
        self.d_step_max = d_step_max
        self.lane_change_penalty = lane_change_penalty
        self.render_mode = render_mode

        # ===========================
        # Internal state
        # ===========================
        self.s = 0.0
        self.d = 0.0
        self.lane_id = 0

        # ===========================
        # Spaces
        # ===========================
        obs_low = np.array([0, -20, 0], dtype=np.float32)
        obs_high = np.array([1, 20, 1], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high)
        self.action_space = spaces.Discrete(2)

        # Render
        self._fig = None
        self._ax = None

        self.in_lane_change = False
        self.target_lane = None


    # ============================================================
    # LOAD YAML
    # ============================================================
    def _load_yaml_obstacles(self, path):

        if not os.path.exists(path):
            print("⚠️ No obstacles.yaml found.")
            return [], [], [], {}

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        road_obstacles = []
        roadside_objects = []
        construction_clusters = []

        # ----------------------
        # ROAD OBSTACLES
        # ----------------------
        for obs in data.get("road_obstacles", []):
            if "lane" in obs:
                lane_id = obs["lane"]
                d = float(self.lane_offsets[lane_id])
            else:
                d = obs.get("d", 0.0)

            road_obstacles.append({
                "type": obs["type"],
                "s": float(obs["s"]),
                "d": float(d),
                "radius": float(obs["radius"]),
            })

        # ----------------------
        # ROADSIDE OBJECTS
        # ----------------------
        for obj in data.get("roadside_objects", []):
            roadside_objects.append({
                "type": obj["type"],
                "s": float(obj["s"]),
                "d": float(obj["d"]),
                "radius": float(obj.get("radius", 0.5)),
                "extra": obj  # store whole block for special logic
            })

        # ----------------------
        # CLUSTERS
        # ----------------------
        for cluster in data.get("construction_zone", []):
            # Expand into multiple cones
            cones = []
            for i in range(cluster["count"]):
                ds = (i - cluster["count"] / 2) * cluster["spread_s"]
                dd = np.random.uniform(-cluster["spread_d"], cluster["spread_d"])
                cones.append({
                    "type": "cone",
                    "s": cluster["center_s"] + ds,
                    "d": cluster["center_d"] + dd,
                    "radius": cluster.get("radius", 0.3)
                })
            construction_clusters.append(cones)

        # ----------------------
        # SCENARIO RANDOMIZATION
        # ----------------------
        scenario_randomization = data.get("scenario", {})

        return road_obstacles, roadside_objects, construction_clusters, scenario_randomization


    # ============================================================
    # RESET
    # ============================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        rng = np.random.default_rng(seed)

        # Initial pose
        self.s = float(rng.uniform(0, 0.05 * self.s_max))
        self.lane_id = int(rng.integers(0, 2))
        self.d = float(self.lane_offsets[self.lane_id])

        return self._get_obs(), self._get_info()


    # ============================================================
    # STEP
    # ============================================================
    def step(self, action):

        reward = 0.0
        
        if not self.in_lane_change:
            # Toggle lane
            #target_lane = self.lane_id
            if action == 1:
                self.in_lane_change = True
                target_lane = 1 - self.lane_id
                #reward -= self.lane_change_penalty # is it necessary?
                target_d = self.lane_offsets[target_lane]
            else:
                # Stay in lane
                target_d = self.lane_offsets[self.lane_id]
        else:
            target_d = self.lane_offsets[self.target_lane]

        # Move laterally
        d_err = target_d - self.d
        step_d = np.clip(d_err, -self.d_step_max, self.d_step_max)
        self.d += step_d


        # Check if lane change is complete
        if self.in_lane_change:
            if abs(self.d - target_d) < 0.2:
                self.lane_id = self.target_lane
                self.in_lane_change = False
                self.target_lane = None

        # Move forward
        s_prev = self.s
        self.s = min(self.s + self.v_ref * self.dt, self.s_max)
        reward += self.s - s_prev

        # Penalize lateral deviation
        reward -= 0.1 * abs(self.d - self.lane_offsets[self.lane_id])

        # Collision
        if self._check_collisions():
            reward -= 50
            terminated = True
        else:
            terminated = bool(self.s >= self.s_max)

        truncated = False

        # Render
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()


    # ============================================================
    # COLLISIONS
    # ============================================================
    def _check_collisions(self):

        # --- Road obstacles ---
        for obs in self.road_obstacles:
            if self._circular_hit(obs):
                return True

        # --- Roadside objects (stop signs, lights) DO NOT COLLIDE ---
        # Later: handle logic for stop violation or red-light violation

        # --- Construction clusters ---
        for cluster in self.construction_clusters:
            for cone in cluster:
                if self._circular_hit(cone):
                    return True

        return False


    def _circular_hit(self, obs):
        ds = self.s - obs["s"]
        dd = self.d - obs["d"]
        return (ds * ds + dd * dd) < (obs["radius"] ** 2)


    # ============================================================
    # OBS + INFO
    # ============================================================
    def _get_obs(self):
        return np.array([
            np.clip(self.s / self.s_max, 0, 1),
            self.d,
            float(self.lane_id)
        ], dtype=np.float32)


    def _get_info(self):
        return {
            "s": self.s,
            "d": self.d,
            "lane": self.lane_id,
        }


    # ============================================================
    # RENDER
    # ============================================================
    def render(self):

        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(9, 6))

        ax = self._ax
        ax.clear()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Lanes
        ax.plot(self.cl_main[:,0], self.cl_main[:,1], "k-", label="Lane 0")
        ax.plot(self.cl_other[:,0], self.cl_other[:,1], "b--", label="Lane 1")

        # Vehicle
        x, y = self.frenet_main.sd_to_xy(self.s, self.d)
        ax.scatter(x, y, c="red", s=70, label="Agent")

        # Draw road obstacles
        for obs in self.road_obstacles:
            ox, oy = self.frenet_main.sd_to_xy(obs["s"], obs["d"])
            circ = plt.Circle((ox, oy), obs["radius"], color="orange", alpha=0.6)
            ax.add_patch(circ)

        # Roadside objects
        for obj in self.roadside_objects:
            ox, oy = self.frenet_main.sd_to_xy(obj["s"], obj["d"])
            circ = plt.Circle((ox, oy), obj["radius"], color="red", alpha=0.6)
            ax.add_patch(circ)
            ax.text(ox, oy + obj["radius"] + 0.3, obj["type"], ha="center")

        # Construction clusters
        for cluster in self.construction_clusters:
            for cone in cluster:
                ox, oy = self.frenet_main.sd_to_xy(cone["s"], cone["d"])
                circ = plt.Circle((ox, oy), cone["radius"], color="yellow", alpha=0.7)
                ax.add_patch(circ)

        ax.set_title(f"s={self.s:.1f}, lane={self.lane_id}")
        ax.legend()
        plt.pause(0.001)

    def close(self):
        if self._fig:
            plt.close(self._fig)
        self._fig = None
