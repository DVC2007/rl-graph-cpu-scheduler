"""Train Attention-Graph + MaskablePPO on SRTFApproxEnv.

Uses:
  - SRTFApproxEnv: 11-feature obs (EWMA, burst std, est. remaining CPU) + SRPT reward shaping
  - AttentionGraphExtractor: dot-product attention aggregation instead of GraphSAGE mean
  - Realistic workload distribution at reset time

Expected to close ~30-40% of the remaining gap to SRTF oracle.
"""
import numpy as np
from sb3_contrib import MaskablePPO

from srpt_approx_env import SRTFApproxEnv
from attention_graph_policy import AttentionGraphExtractor
from realistic_workload import generate_workload


class RealisticSRTFEnv(SRTFApproxEnv):
    """SRTFApproxEnv with realistic burst distributions at each reset."""

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=None)
        n = self.active_processes
        wl = generate_workload(seed, num_processes=n)
        self.processes = []
        for i, item in enumerate(wl):
            cpu_bursts = [int(v) for v in item["cpu_bursts"]]
            io_bursts  = [int(v) for v in item["io_bursts"]]
            self.processes.append({
                "name": f"P{i+1}",
                "cpu_bursts": cpu_bursts, "io_bursts": io_bursts,
                "current_burst_index": 0, "remaining_in_burst": cpu_bursts[0],
                "current_io_index": 0, "remaining_io": 0,
                "waiting": 0, "type": str(item["type"]),
                "cpu_used": 0, "times_scheduled": 0,
                "completed_bursts": [], "last_burst": 0, "average_burst": 0,
                "ready": True, "blocked_time": 0, "finished": False,
                "ewma_burst": 0.0, "burst_std": 0.0,
            })
        self.current_time = 0
        return self._get_observation(), {"action_mask": self.action_masks()}


env = RealisticSRTFEnv(
    num_processes=8,
    random_process_count=True,
    starvation_threshold=10,
    aging_factor=0.5,
)

policy_kwargs = {
    "features_extractor_class":  AttentionGraphExtractor,
    "features_extractor_kwargs": {"features_dim": 64, "max_processes": 8},
    "net_arch": {"pi": [64, 32], "vf": [64, 32]},
}

model = MaskablePPO(
    "MlpPolicy", env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=2048,
    seed=42,
    verbose=1,
)
model.learn(total_timesteps=200_000)
model.save("attention_graph_srpt_scheduler")
print("Saved: attention_graph_srpt_scheduler.zip")
