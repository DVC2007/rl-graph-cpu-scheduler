"""Full ablation benchmark: all schedulers × both workload distributions.

Runs 500 episodes per (scheduler, workload_type) combination and produces:
  results/ablation_summary.csv   — one row per (scheduler, workload_type)
  results/ablation_raw.csv       — every episode
  Printed table for the paper

Workload types
--------------
uniform   : uniform-random bursts (original benchmark, seeds 10000-10499)
realistic : exponential burst distributions (Coffman-Kleinrock model, seeds 20000-20499)

Schedulers
----------
Classical (no oracle): FCFS, Round Robin, Priority+Aging, MLFQ, CFS, Stride
Oracle lower bound:    SRTF  (uses hidden burst lengths — reported for reference only)
RL (no oracle):        Plain MaskablePPO, GraphSAGE-RL (uniform), GraphSAGE-RL (realistic)
"""
import csv
import os
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO

from rl_env import CPUSchedulingEnv, PROCESS_WAIT_WEIGHT
from scheduler_baselines import (
    FCFSPolicy, RoundRobinPolicy, PriorityAgingPolicy, MLFQPolicy,
    CFSPolicy, SRTFPolicy, StridePolicy,
)
from realistic_workload import generate_workload

ROOT = os.path.dirname(__file__)
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

N = int(os.environ.get("RL_BENCHMARK_WORKLOADS", "500"))
THRESHOLD = 10
AGING = 0.5
BOOTSTRAP = 2_000


# ---------------------------------------------------------------------------
# Workload factories
# ---------------------------------------------------------------------------

def uniform_workload(seed):
    """Original uniform-random burst workload (seeded via env)."""
    return None  # env generates it internally


def realistic_workload(seed):
    """Exponential burst workload from Coffman-Kleinrock model."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 9))
    return generate_workload(seed, num_processes=n, rng=rng)


WORKLOAD_FACTORIES = {
    "uniform":   (10_000, uniform_workload),
    "realistic": (20_000, realistic_workload),
}


# ---------------------------------------------------------------------------
# Policy wrappers
# ---------------------------------------------------------------------------

class PPOPolicy:
    def __init__(self, model):
        self.model = model

    def act(self, env):
        action, _ = self.model.predict(
            env._get_observation(), deterministic=True, action_masks=env.action_masks()
        )
        return int(action)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(policy, seed, workload):
    if workload is None:
        n = None
        env = CPUSchedulingEnv(
            num_processes=8, random_process_count=True,
            starvation_threshold=THRESHOLD, aging_factor=AGING,
        )
        env.reset(seed=seed)
    else:
        n = len(workload)
        env = CPUSchedulingEnv(
            num_processes=n, starvation_threshold=THRESHOLD, aging_factor=AGING,
        )
        env.reset(seed=seed, options={"workload": workload})

    while not all(p["finished"] for p in env.processes):
        env.step(policy.act(env))

    waits = [p["waiting"] for p in env.processes]
    by_type = defaultdict(list)
    for p in env.processes:
        by_type[p["type"]].append(p["waiting"])

    row = {
        "average_waiting":          float(np.mean(waits)),
        "priority_weighted_waiting": float(np.mean(
            [PROCESS_WAIT_WEIGHT[p["type"]] * p["waiting"] for p in env.processes]
        )),
        "episode_max_waiting":      max(waits),
        "completion_time":          env.current_time,
        "starvation_count":         sum(w > THRESHOLD for w in waits),
        "process_count":            len(waits),
        "p95_waiting":              float(np.percentile(waits, 95)),
    }
    for kind in ("SYSTEM", "INTERACTIVE", "BACKGROUND"):
        row[f"{kind}_waiting_sum"] = sum(by_type[kind])
        row[f"{kind}_count"]       = len(by_type[kind])
    return row


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mean_ci(values, rng):
    arr = np.asarray(values, dtype=float)
    boot = np.mean(rng.choice(arr, size=(BOOTSTRAP, len(arr)), replace=True), axis=1)
    return float(np.mean(arr)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize(rows, rng):
    total = sum(r["process_count"] for r in rows)
    m = {}
    m["average_waiting"], m["aw_ci_low"], m["aw_ci_high"] = mean_ci(
        [r["average_waiting"] for r in rows], rng
    )
    m["priority_weighted_waiting"] = (
        sum(r["priority_weighted_waiting"] * r["process_count"] for r in rows) / total
    )
    m["mean_episode_max"] = float(np.mean([r["episode_max_waiting"] for r in rows]))
    m["starvation_rate"] = (
        sum(r["starvation_count"] for r in rows) / sum(r["process_count"] for r in rows)
    )
    m["p95_waiting"] = float(np.mean([r["p95_waiting"] for r in rows]))
    for kind in ("SYSTEM", "INTERACTIVE", "BACKGROUND"):
        s = sum(r[f"{kind}_waiting_sum"] for r in rows)
        c = sum(r[f"{kind}_count"] for r in rows)
        m[f"{kind}_avg_waiting"] = s / c if c else 0.0
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load RL models
    rl_models = {}
    for path, label in [
        ("graphsage_maskable_ppo_variable_scheduler", "GraphSAGE-RL (uniform)"),
        ("graphsage_realistic_scheduler",             "GraphSAGE-RL (realistic)"),
        ("maskable_ppo_variable_scheduler",           "Plain MaskablePPO"),
    ]:
        try:
            rl_models[label] = MaskablePPO.load(os.path.join(ROOT, path))
            print(f"Loaded {path}")
        except Exception:
            print(f"Skipped (not found): {path}")

    POLICIES = {
        # Classical — no oracle
        "FCFS":              lambda: FCFSPolicy(),
        "Round Robin (q=2)": lambda: RoundRobinPolicy(quantum=2),
        "Priority + Aging":  lambda: PriorityAgingPolicy(aging_factor=AGING),
        "MLFQ":              lambda: MLFQPolicy(),
        "CFS":               lambda: CFSPolicy(),
        "Stride":            lambda: StridePolicy(),
        # Oracle lower bound
        "SRTF (oracle)":     lambda: SRTFPolicy(),
    }
    for label, m in rl_models.items():
        POLICIES[label] = (lambda _m: lambda: PPOPolicy(_m))(m)

    rng = np.random.default_rng(2026)
    all_rows = []
    summary_rows = []

    for wl_name, (base_seed, wl_factory) in WORKLOAD_FACTORIES.items():
        print(f"\n=== Workload: {wl_name} ===")
        workloads = [wl_factory(base_seed + i) for i in range(N)]

        for policy_name, factory in POLICIES.items():
            print(f"  {policy_name}...")
            rows = []
            for i in range(N):
                seed = base_seed + i
                row = run_episode(factory(), seed, workloads[i])
                row.update(model=policy_name, workload=wl_name, seed=seed)
                rows.append(row)
            all_rows.extend(rows)

            s = summarize(rows, rng)
            s.update(model=policy_name, workload=wl_name)
            summary_rows.append(s)

    # Save raw
    with open(os.path.join(OUT, "ablation_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader(); writer.writerows(all_rows)

    # Save summary
    with open(os.path.join(OUT, "ablation_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader(); writer.writerows(summary_rows)

    # Print paper-ready table
    col_order = list(POLICIES.keys())
    for wl_name in ("uniform", "realistic"):
        print(f"\n{'─'*85}")
        print(f"  Workload: {wl_name}  (n={N} episodes, 95% CI via bootstrap)")
        print(f"{'─'*85}")
        print(f"  {'Scheduler':<28} {'Avg Wait':>10}  {'95% CI':>19}  {'Starvation':>12}  {'P-Wtd Wait':>12}")
        print(f"  {'─'*28} {'─'*10}  {'─'*19}  {'─'*12}  {'─'*12}")
        for name in col_order:
            s = next(r for r in summary_rows if r["model"] == name and r["workload"] == wl_name)
            ci = f"[{s['aw_ci_low']:.2f}, {s['aw_ci_high']:.2f}]"
            oracle = "†" if "oracle" in name.lower() else " "
            print(f"  {name+oracle:<28} {s['average_waiting']:>10.2f}  {ci:>19}  {s['starvation_rate']:>12.1%}  {s['priority_weighted_waiting']:>12.2f}")
        print()
        # vs best classical
        classical = [r for r in summary_rows if r["workload"] == wl_name
                     and r["model"] not in ("SRTF (oracle)",) and "RL" not in r["model"]
                     and "MaskablePPO" not in r["model"]]
        best_cl = min(classical, key=lambda r: r["average_waiting"])
        best_rl = min(
            [r for r in summary_rows if r["workload"] == wl_name and ("RL" in r["model"] or "MaskablePPO" in r["model"])],
            key=lambda r: r["average_waiting"],
        )
        oracle_row = next(r for r in summary_rows if r["model"] == "SRTF (oracle)" and r["workload"] == wl_name)
        gap_to_classical = best_cl["average_waiting"] - best_rl["average_waiting"]
        gap_to_oracle = best_rl["average_waiting"] - oracle_row["average_waiting"]
        pct = gap_to_classical / best_cl["average_waiting"] * 100
        print(f"  Best classical: {best_cl['model']} ({best_cl['average_waiting']:.2f})")
        print(f"  Best RL:        {best_rl['model']} ({best_rl['average_waiting']:.2f})")
        print(f"  RL improvement over best classical: {gap_to_classical:.2f} units ({pct:.1f}%)")
        print(f"  Gap to oracle (SRTF): {gap_to_oracle:.2f} units")

    print("\n† SRTF uses hidden future burst lengths — oracle lower bound only, not a fair comparison.")
    print(f"\nSaved: results/ablation_summary.csv, results/ablation_raw.csv")


if __name__ == "__main__":
    main()
