"""Train GraphSAGE + MaskablePPO on realistic workload distributions.

Uses exponential CPU/IO burst distributions (Coffman-Kleinrock model) instead
of uniform random, giving the policy experience with realistic type-specific
burst patterns (short SYSTEM bursts, long BACKGROUND bursts, etc.).
"""
import numpy as np
from sb3_contrib import MaskablePPO
from gymnasium import Env
import gymnasium as gym
from gymnasium import spaces

from rl_env import CPUSchedulingEnv
from graph_policy import GraphSAGEFeatureExtractor
from realistic_workload import generate_workload


class RealisticCPUEnv(CPUSchedulingEnv):
    """CPUSchedulingEnv that replaces uniform-random bursts with realistic distributions."""

    def reset(self, seed=None, options=None):
        # Let the parent complete its reset (seeds np_random, sets active_processes)
        obs, info = super().reset(seed=seed, options=None)
        n = self.active_processes
        # Generate a realistic workload matching the already-determined process count
        wl = generate_workload(seed, num_processes=n)
        # Replace processes in-place; mirror the dict structure from rl_env.py
        self.processes = []
        for i, item in enumerate(wl):
            cpu_bursts = [int(v) for v in item["cpu_bursts"]]
            io_bursts  = [int(v) for v in item["io_bursts"]]
            self.processes.append({
                "name": f"P{i+1}",
                "cpu_bursts": cpu_bursts,
                "io_bursts": io_bursts,
                "current_burst_index": 0,
                "remaining_in_burst": cpu_bursts[0],
                "current_io_index": 0,
                "remaining_io": 0,
                "waiting": 0,
                "type": str(item["type"]),
                "cpu_used": 0,
                "times_scheduled": 0,
                "completed_bursts": [],
                "last_burst": 0,
                "average_burst": 0,
                "ready": True,
                "blocked_time": 0,
                "finished": False,
            })
        self.current_time = 0
        return self._get_observation(), {"action_mask": self.action_masks()}


env = RealisticCPUEnv(
    num_processes=8,
    random_process_count=True,
    starvation_threshold=10,
    aging_factor=0.5,
)

policy_kwargs = {
    "features_extractor_class": GraphSAGEFeatureExtractor,
    "features_extractor_kwargs": {"features_dim": 48, "max_processes": 8},
    "net_arch": {"pi": [64], "vf": [64]},
}

model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, seed=42, verbose=1)
model.learn(total_timesteps=150_000)
model.save("graphsage_realistic_scheduler")
print("Saved: graphsage_realistic_scheduler.zip")
