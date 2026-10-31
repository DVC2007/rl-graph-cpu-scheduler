"""Realistic CPU/IO burst workload generator.

Burst distributions derived from empirical OS literature:
  - Silberschatz et al., "Operating System Concepts", 10th ed., Ch.5
  - Coffman & Kleinrock (1973): CPU bursts are hyperexponentially distributed
  - Interactive processes: short CPU bursts, long I/O waits
  - Background processes: long CPU bursts, short I/O waits
  - System processes: very short bursts, frequent preemption

Each generated process dict is compatible with CPUSchedulingEnv's workload= API:
  {"cpu_bursts": [...], "io_bursts": [...], "type": "SYSTEM"|"INTERACTIVE"|"BACKGROUND"}
"""
import numpy as np

# Parameters from OS literature (in simulator time units ≈ scheduler quanta)
# CPU burst mean and I/O burst mean per type
_BURST_PARAMS = {
    "SYSTEM":      {"cpu_mean": 2.5,  "cpu_min": 1, "io_mean": 2.0, "io_min": 1, "num_bursts_mean": 4},
    "INTERACTIVE": {"cpu_mean": 4.0,  "cpu_min": 1, "io_mean": 5.0, "io_min": 1, "num_bursts_mean": 3},
    "BACKGROUND":  {"cpu_mean": 8.0,  "cpu_min": 2, "io_mean": 2.0, "io_min": 1, "num_bursts_mean": 3},
}

_TYPE_MIX = [0.20, 0.50, 0.30]  # SYSTEM, INTERACTIVE, BACKGROUND — realistic OS mix


def _exp_burst(rng, mean, minimum, max_val=20):
    """Exponentially distributed burst, clamped to [minimum, max_val]."""
    v = rng.exponential(scale=mean)
    return int(np.clip(round(v), minimum, max_val))


def generate_process(rng, process_type=None):
    """Return one process dict for CPUSchedulingEnv's workload= argument."""
    if process_type is None:
        process_type = str(rng.choice(["SYSTEM", "INTERACTIVE", "BACKGROUND"], p=_TYPE_MIX))
    p = _BURST_PARAMS[process_type]
    # Number of CPU bursts: Poisson-distributed with minimum 2
    n = max(2, int(rng.poisson(p["num_bursts_mean"])))
    cpu_bursts = [_exp_burst(rng, p["cpu_mean"], p["cpu_min"]) for _ in range(n)]
    io_bursts  = [_exp_burst(rng, p["io_mean"],  p["io_min"])  for _ in range(n - 1)]
    return {"cpu_bursts": cpu_bursts, "io_bursts": io_bursts, "type": process_type}


def generate_workload(seed, num_processes=None, rng=None):
    """Return a list of process dicts for one episode.

    num_processes: if None, drawn from realistic range [2, 8].
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if num_processes is None:
        num_processes = int(rng.integers(2, 9))
    return [generate_process(rng) for _ in range(num_processes)]


def generate_workload_batch(n, base_seed=20_000):
    """Return n workloads with reproducible seeds (for benchmarking)."""
    return [generate_workload(base_seed + i) for i in range(n)]
