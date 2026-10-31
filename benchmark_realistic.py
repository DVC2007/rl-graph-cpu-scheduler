"""Benchmark all schedulers on realistic (exponential burst) workloads.

Generates 500 seeded episodes using the Coffman-Kleinrock burst model and
compares FCFS, Round Robin, Priority+Aging, MLFQ, and both RL models.
Results saved to results/realistic_summary.csv and results/realistic_raw.csv.
"""
import csv
import os
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO

from rl_env import CPUSchedulingEnv, PROCESS_WAIT_WEIGHT
from scheduler_baselines import FCFSPolicy, RoundRobinPolicy, PriorityAgingPolicy, MLFQPolicy
from realistic_workload import generate_workload

ROOT = os.path.dirname(__file__)
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

N = int(os.environ.get("RL_BENCHMARK_WORKLOADS", "500"))
BASE_SEED = 20_000
THRESHOLD = 10
AGING = 0.5
BOOTSTRAP = 2_000


class PPOPolicy:
    def __init__(self, model):
        self.model = model

    def act(self, env):
        action, _ = self.model.predict(
            env._get_observation(), deterministic=True, action_masks=env.action_masks()
        )
        return int(action)


def run_episode(policy, workload):
    n = len(workload)
    env = CPUSchedulingEnv(num_processes=n, starvation_threshold=THRESHOLD, aging_factor=AGING)
    env.reset(seed=None, options={"workload": workload})
    while not all(p["finished"] for p in env.processes):
        env.step(policy.act(env))
    waits = [p["waiting"] for p in env.processes]
    by_type = defaultdict(list)
    for p in env.processes:
        by_type[p["type"]].append(p["waiting"])
    row = {
        "average_waiting": float(np.mean(waits)),
        "priority_weighted_waiting": float(
            np.mean([PROCESS_WAIT_WEIGHT[p["type"]] * p["waiting"] for p in env.processes])
        ),
        "episode_max_waiting": max(waits),
        "completion_time": env.current_time,
        "starvation_count": sum(w > THRESHOLD for w in waits),
        "process_count": len(waits),
    }
    for kind in ("SYSTEM", "INTERACTIVE", "BACKGROUND"):
        row[f"{kind}_waiting_sum"] = sum(by_type[kind])
        row[f"{kind}_count"] = len(by_type[kind])
    return row


def mean_ci(values, rng):
    arr = np.asarray(values, dtype=float)
    boot = np.mean(rng.choice(arr, size=(BOOTSTRAP, len(arr)), replace=True), axis=1)
    return float(np.mean(arr)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def evaluate(name, policy_factory, workloads):
    rows = []
    for i, wl in enumerate(workloads):
        row = run_episode(policy_factory(), wl)
        row.update(model=name, workload_idx=BASE_SEED + i)
        rows.append(row)
    return rows


def summarize(rows, model_names):
    rng = np.random.default_rng(2026)
    summaries = []
    for model in model_names:
        vals = [r for r in rows if r["model"] == model]
        total = sum(r["process_count"] for r in vals)
        m = {"model": model}
        m["average_waiting"], m["aw_ci_low"], m["aw_ci_high"] = mean_ci(
            [r["average_waiting"] for r in vals], rng
        )
        m["priority_weighted_waiting"] = (
            sum(r["priority_weighted_waiting"] * r["process_count"] for r in vals) / total
        )
        m["mean_episode_max"] = float(np.mean([r["episode_max_waiting"] for r in vals]))
        m["starvation_rate"] = (
            sum(r["starvation_count"] for r in vals) /
            sum(r["process_count"] for r in vals)
        )
        summaries.append(m)
    return summaries


def main():
    print(f"Generating {N} realistic workloads...")
    workloads = [generate_workload(BASE_SEED + i) for i in range(N)]

    models = {}
    for path in ["graphsage_maskable_ppo_variable_scheduler", "graphsage_realistic_scheduler",
                 "maskable_ppo_variable_scheduler"]:
        try:
            models[path] = MaskablePPO.load(os.path.join(ROOT, path))
            print(f"Loaded {path}")
        except Exception:
            print(f"Skipped (not found): {path}")

    POLICIES = {
        "FCFS":              lambda: FCFSPolicy(),
        "Round Robin (q=2)": lambda: RoundRobinPolicy(quantum=2),
        "Priority + Aging":  lambda: PriorityAgingPolicy(aging_factor=AGING),
        "MLFQ":              lambda: MLFQPolicy(),
    }
    for name, m in models.items():
        label = "GraphSAGE-RL (realistic)" if "realistic" in name else \
                "GraphSAGE-RL (uniform)" if "graphsage" in name else "Plain MaskablePPO"
        POLICIES[label] = (lambda _m: lambda: PPOPolicy(_m))(m)

    all_rows = []
    for name, factory in POLICIES.items():
        print(f"  Evaluating {name}...")
        all_rows.extend(evaluate(name, factory, workloads))

    with open(os.path.join(OUT, "realistic_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    summaries = summarize(all_rows, list(POLICIES.keys()))
    with open(os.path.join(OUT, "realistic_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print("\n=== Realistic Workload Benchmark ===")
    print(f"{'Model':<30} {'Avg Wait':>10} {'CI':>18} {'Starvation':>12}")
    print("-" * 75)
    for s in summaries:
        ci = f"[{s['aw_ci_low']:.2f}, {s['aw_ci_high']:.2f}]"
        print(f"{s['model']:<30} {s['average_waiting']:>10.2f} {ci:>18} {s['starvation_rate']:>12.1%}")

    print(f"\nSaved: results/realistic_summary.csv, results/realistic_raw.csv")


if __name__ == "__main__":
    main()
