"""SRPT-Approximating CPU Scheduling Environment.

Extends CPUSchedulingEnv with three improvements to close the gap to SRTF oracle:

1. EWMA burst prediction (α=0.5) — classic SJF approximation from OS textbooks.
   τ_{n+1} = α·t_n + (1−α)·τ_{n-1}. Better predictor of next burst than simple mean.
   Reference: Silberschatz et al., "Operating System Concepts", 10th ed., Ch.5.

2. Burst standard deviation — captures predictability; high std → less trustworthy EWMA.

3. SRPT-shaped reward bonus — small bonus for scheduling the process with the shortest
   estimated remaining CPU time, nudging the policy toward SRTF-like prioritization
   without oracle knowledge.

Observation: 11 features per process (vs 8 in base env).
  0: waiting time
  1: type code
  2: cpu_used
  3: times_scheduled
  4: last_burst
  5: average_burst  (arithmetic mean, same as base env)
  6: ready_flag
  7: blocked_time
  8: ewma_burst     ← NEW: EWMA(α=0.5) of completed burst durations
  9: burst_std      ← NEW: std-dev of completed bursts (0 if <2 bursts)
 10: est_remaining  ← NEW: ewma_burst × max(1, estimated_remaining_bursts)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl_env import (
    CPUSchedulingEnv, MAX_PROCESSES, PROCESS_TYPE_VALUE,
    PROCESS_WAIT_WEIGHT, PROCESS_TYPES, STARVATION_THRESHOLD,
    AGING_FACTOR, IDLE_ACTION,
)

EWMA_ALPHA   = 0.5
SRPT_WEIGHT  = 0.3   # weight of SRPT bonus relative to waiting penalty


class SRTFApproxEnv(CPUSchedulingEnv):
    """CPUSchedulingEnv extended with burst-prediction features and SRPT reward shaping."""

    FEATURES_PER_PROCESS = 11   # 8 base + ewma, burst_std, est_remaining

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override observation space for 11 features per slot
        self.features_per_process = self.FEATURES_PER_PROCESS
        self.observation_space = spaces.Box(
            low=0,
            high=np.finfo(np.float32).max,
            shape=(MAX_PROCESSES * self.FEATURES_PER_PROCESS,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Initialize EWMA = first burst length (prior = first burst, observable after reset)
        for p in self.processes:
            p["ewma_burst"]   = float(p["cpu_bursts"][0])   # prior from first visible burst length? No — use type prior
            p["burst_std"]    = 0.0
        # Actually the first burst length IS observable indirectly (remaining_in_burst shows it),
        # but to stay strict we initialise EWMA to 0 (no history) and update after first completion.
        for p in self.processes:
            p["ewma_burst"]   = 0.0
            p["burst_std"]    = 0.0
        return self._get_observation(), info

    # ------------------------------------------------------------------
    # Override step to (a) update EWMA after burst completion and
    # (b) add SRPT-shaped reward bonus.
    # ------------------------------------------------------------------

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Update EWMA and burst_std for the just-run process
        if action != IDLE_ACTION and action < len(self.processes):
            p = self.processes[action]
            if p["completed_bursts"]:
                last = float(p["last_burst"])
                old_ewma = p.get("ewma_burst", last)
                new_ewma = EWMA_ALPHA * last + (1 - EWMA_ALPHA) * old_ewma
                p["ewma_burst"] = new_ewma
                # Welford-style std update using completed_bursts list
                bursts = p["completed_bursts"]
                p["burst_std"] = float(np.std(bursts)) if len(bursts) >= 2 else 0.0

        # SRPT reward shaping: bonus for choosing the ready process with
        # the minimum estimated remaining CPU time.
        if action != IDLE_ACTION and action < len(self.processes):
            ready = [i for i, q in enumerate(self.processes)
                     if q["ready"] and not q["finished"]]
            ready_with_current = ready + ([action] if action not in ready else [])
            if len(ready_with_current) > 1:
                ewmas = [self._est_remaining(i) for i in ready_with_current]
                min_idx = ready_with_current[ewmas.index(min(ewmas))]
                max_est  = max(ewmas)
                # Positive bonus proportional to how much shorter this process is vs average
                my_est  = self._est_remaining(action)
                avg_est = float(np.mean(ewmas))
                if avg_est > 0:
                    srpt_bonus = SRPT_WEIGHT * (avg_est - my_est) / (avg_est + 1e-6)
                    reward += srpt_bonus

        return self._get_observation(), reward, terminated, truncated, info

    def _est_remaining(self, idx):
        """Estimated remaining CPU for process idx using EWMA."""
        p = self.processes[idx]
        ewma = p.get("ewma_burst", 0.0)
        if ewma == 0.0:
            # No history: fall back to last_burst or a type-based prior
            ewma = float(p["last_burst"]) if p["last_burst"] > 0 else {
                "SYSTEM": 2.5, "INTERACTIVE": 4.0, "BACKGROUND": 8.0,
            }[p["type"]]
        # Estimate remaining bursts from times_scheduled (heuristic: avg 3 total bursts)
        completed = max(p["times_scheduled"], 1)
        estimated_remaining_bursts = max(1.0, 3.0 - completed)
        return ewma * estimated_remaining_bursts

    def _get_observation(self):
        obs = []
        for p in self.processes:
            if p["finished"]:
                obs.extend([0.0] * self.FEATURES_PER_PROCESS)
            else:
                ewma    = p.get("ewma_burst", 0.0)
                burst_std = p.get("burst_std", 0.0)
                est_rem = self._est_remaining(self.processes.index(p))
                obs.extend([
                    p["waiting"],
                    PROCESS_TYPE_VALUE[p["type"]],
                    p["cpu_used"],
                    p["times_scheduled"],
                    p["last_burst"],
                    p["average_burst"],
                    1.0 if p["ready"] else 0.0,
                    p["blocked_time"],
                    ewma,        # feature 8
                    burst_std,   # feature 9
                    est_rem,     # feature 10
                ])
        obs.extend([0.0] * ((MAX_PROCESSES - len(self.processes)) * self.FEATURES_PER_PROCESS))
        return np.array(obs, dtype=np.float32)
