"""Reproducible, paired benchmark for classical and RL CPU schedulers."""
import csv
import os
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO

from rl_env import CPUSchedulingEnv, PROCESS_WAIT_WEIGHT
from scheduler_baselines import FCFSPolicy, PriorityAgingPolicy, RoundRobinPolicy

ROOT = os.path.dirname(__file__)
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
N = int(os.environ.get("RL_BENCHMARK_WORKLOADS", "500"))
THRESHOLD = 10
AGING_FACTOR = 0.5
BOOTSTRAP_SAMPLES = 2_000


class PPOPolicy:
    def __init__(self, model):
        self.model = model

    def act(self, env):
        action, _ = self.model.predict(env._get_observation(), deterministic=True, action_masks=env.action_masks())
        return int(action)


def run(policy, seed):
    env = CPUSchedulingEnv(num_processes=8, random_process_count=True, starvation_threshold=THRESHOLD, aging_factor=AGING_FACTOR)
    env.reset(seed=seed)
    while not all(p["finished"] for p in env.processes):
        env.step(policy.act(env))
    waits = [p["waiting"] for p in env.processes]
    by_type = defaultdict(list)
    for process in env.processes:
        by_type[process["type"]].append(process["waiting"])
    row = {
        "average_waiting": float(np.mean(waits)),
        "priority_weighted_waiting": float(np.mean([PROCESS_WAIT_WEIGHT[p["type"]] * p["waiting"] for p in env.processes])),
        "episode_max_waiting": max(waits),
        "completion_time": env.current_time,
        "starvation_count": sum(wait > THRESHOLD for wait in waits),
        "process_count": len(waits),
        "wait_p95": float(np.percentile(waits, 95)),
        "wait_p99": float(np.percentile(waits, 99)),
    }
    for kind in ("SYSTEM", "INTERACTIVE", "BACKGROUND"):
        row[f"{kind}_waiting_sum"] = sum(by_type[kind])
        row[f"{kind}_count"] = len(by_type[kind])
    return row


def evaluate(name, policy_factory):
    rows = []
    for index in range(N):
        seed = 10_000 + index
        row = run(policy_factory(seed), seed)
        row.update(model=name, aging_factor=AGING_FACTOR, workload_seed=seed)
        rows.append(row)
    return rows


def mean_ci(values, rng):
    values = np.asarray(values, dtype=float)
    boot = np.mean(rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True), axis=1)
    return float(np.mean(values)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize(rows, model_names):
    rng = np.random.default_rng(2026)
    summaries = []
    for model in model_names:
        values = [row for row in rows if row["model"] == model]
        total_processes = sum(row["process_count"] for row in values)
        item = {"model": model, "aging_factor": AGING_FACTOR}
        item["average_waiting"] = sum(row["average_waiting"] * row["process_count"] for row in values) / total_processes
        item["priority_weighted_waiting"] = sum(row["priority_weighted_waiting"] * row["process_count"] for row in values) / total_processes
        item["mean_episode_max_waiting"] = float(np.mean([row["episode_max_waiting"] for row in values]))
        item["global_max_waiting"] = max(row["episode_max_waiting"] for row in values)
        item["completion_time"] = float(np.mean([row["completion_time"] for row in values]))
        item["starvation_rate"] = sum(row["starvation_count"] for row in values) / total_processes
        item["p95_waiting"] = float(np.mean([row["wait_p95"] for row in values]))
        item["p99_waiting"] = float(np.mean([row["wait_p99"] for row in values]))
        for kind in ("SYSTEM", "INTERACTIVE", "BACKGROUND"):
            count = sum(row[f"{kind}_count"] for row in values)
            item[f"{kind}_waiting"] = sum(row[f"{kind}_waiting_sum"] for row in values) / count if count else 0.0
        weighted = sum(item[f"{kind}_waiting"] * sum(row[f"{kind}_count"] for row in values) for kind in ("SYSTEM", "INTERACTIVE", "BACKGROUND")) / total_processes
        assert np.isclose(weighted, item["average_waiting"]), f"type-weighted waiting mismatch for {model}"
        mean, low, high = mean_ci([row["average_waiting"] for row in values], rng)
        item.update(episode_average_waiting=mean, average_waiting_ci_low=low, average_waiting_ci_high=high)
        summaries.append(item)
    return summaries


def paired_differences(rows, final_model, model_names):
    by_model = {name: {row["workload_seed"]: row for row in rows if row["model"] == name} for name in model_names}
    final = by_model[final_model]
    rng = np.random.default_rng(2027)
    output = []
    for baseline in model_names:
        if baseline == final_model:
            continue
        differences = [final[seed]["average_waiting"] - by_model[baseline][seed]["average_waiting"] for seed in final]
        mean, low, high = mean_ci(differences, rng)
        output.append({"comparison": f"{final_model} - {baseline}", "metric": "episode_average_waiting", "mean_difference": mean, "ci_95_low": low, "ci_95_high": high, "graph_better": high < 0})
    return output


def main():
    specs = [
        ("FCFS", lambda seed: FCFSPolicy()),
        ("Round Robin (q=2)", lambda seed: RoundRobinPolicy(quantum=2)),
        ("Priority + aging", lambda seed: PriorityAgingPolicy(AGING_FACTOR)),
    ]
    for checkpoint, name in (("maskable_ppo_variable_scheduler", "Plain MaskablePPO"), ("graphsage_maskable_ppo_variable_scheduler", "GraphSAGE + MaskablePPO")):
        path = os.path.join(ROOT, checkpoint + ".zip")
        if os.path.exists(path):
            model = MaskablePPO.load(path)
            specs.append((name, lambda seed, model=model: PPOPolicy(model)))
        else:
            print(f"Skipping missing checkpoint: {path}")

    rows = []
    for name, factory in specs:
        rows.extend(evaluate(name, factory))
    names = [name for name, _ in specs]
    summary_rows = summarize(rows, names)
    comparisons = paired_differences(rows, "GraphSAGE + MaskablePPO", names)
    for filename, fieldnames, data in (("raw_results.csv", list(rows[0]), rows), ("summary_results.csv", list(summary_rows[0]), summary_rows), ("paired_average_waiting.csv", list(comparisons[0]), comparisons)):
        with open(os.path.join(OUT, filename), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    print(f"\nBenchmark: {N} identical seeded workloads; aging_factor={AGING_FACTOR}")
    print("model | average waiting | priority-weighted waiting | mean episode maximum | global maximum | completion time | starvation rate | SYSTEM | INTERACTIVE | BACKGROUND")
    for row in summary_rows:
        print(f"{row['model']} | {row['average_waiting']:.3f} | {row['priority_weighted_waiting']:.3f} | {row['mean_episode_max_waiting']:.3f} | {row['global_max_waiting']:.3f} | {row['completion_time']:.3f} | {row['starvation_rate']:.3%} | {row['SYSTEM_waiting']:.3f} | {row['INTERACTIVE_waiting']:.3f} | {row['BACKGROUND_waiting']:.3f}")
    print("Wrote results/raw_results.csv, results/summary_results.csv, and results/paired_average_waiting.csv")


if __name__ == "__main__":
    main()
