"""Evaluate five independently trained Plain-RL and Graph-RL checkpoints."""
import argparse
import csv
import os

import numpy as np
from sb3_contrib import MaskablePPO

from research_benchmark import AGING_FACTOR, N, PPOPolicy, run

ROOT = os.path.dirname(__file__)
OUT = os.path.join(ROOT, "results")
CHECKPOINTS = os.path.join(OUT, "checkpoints")
AGENTS = os.path.join(OUT, "multi_seed_agents.csv")
SUMMARY = os.path.join(OUT, "multi_seed_summary.csv")
PAIRED = os.path.join(OUT, "multi_seed_paired_average_waiting.csv")
VS_FCFS = os.path.join(OUT, "multi_seed_graph_vs_fcfs.csv")
SEEDS = (11, 22, 33, 44, 55)


def agent_summary(rows, label, seed):
    total = sum(row["process_count"] for row in rows)
    return {
        "model": label, "training_seed": seed,
        "average_waiting": sum(row["average_waiting"] * row["process_count"] for row in rows) / total,
        "priority_weighted_waiting": sum(row["priority_weighted_waiting"] * row["process_count"] for row in rows) / total,
        "mean_episode_max_waiting": float(np.mean([row["episode_max_waiting"] for row in rows])),
        "completion_time": float(np.mean([row["completion_time"] for row in rows])),
        "starvation_rate": sum(row["starvation_count"] for row in rows) / total,
    }


def bootstrap_ci(values, rng):
    values = np.asarray(values, dtype=float)
    samples = np.mean(rng.choice(values, size=(10_000, len(values)), replace=True), axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def evaluate_one(kind, seed):
    label = "Plain MaskablePPO" if kind == "plain" else "GraphSAGE + MaskablePPO"
    path = os.path.join(CHECKPOINTS, f"{kind}_seed_{seed}.zip")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model = MaskablePPO.load(path)
    row = agent_summary([run(PPOPolicy(model), 10_000 + index) for index in range(N)], label, seed)
    existing = []
    if os.path.exists(AGENTS):
        with open(AGENTS, newline="") as handle:
            existing = list(csv.DictReader(handle))
    existing = [item for item in existing if not (item["model"] == label and int(item["training_seed"]) == seed)]
    existing.append(row)
    with open(AGENTS, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerows(existing)
    print(f"Saved {label}, seed {seed}: average waiting {row['average_waiting']:.3f}")


def summarize():
    with open(AGENTS, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10:
        raise RuntimeError(f"Expected 10 agent rows; found {len(rows)}")
    rng = np.random.default_rng(2028)
    output = []
    for label in ("Plain MaskablePPO", "GraphSAGE + MaskablePPO"):
        agents = [row for row in rows if row["model"] == label]
        result = {"model": label, "training_seeds": len(agents), "aging_factor": AGING_FACTOR}
        for metric in ("average_waiting", "priority_weighted_waiting", "mean_episode_max_waiting", "completion_time", "starvation_rate"):
            values = [float(agent[metric]) for agent in agents]
            low, high = bootstrap_ci(values, rng)
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        output.append(result)
    with open(SUMMARY, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    plain = {int(row["training_seed"]): float(row["average_waiting"]) for row in rows if row["model"] == "Plain MaskablePPO"}
    graph = {int(row["training_seed"]): float(row["average_waiting"]) for row in rows if row["model"] == "GraphSAGE + MaskablePPO"}
    differences = [graph[seed] - plain[seed] for seed in SEEDS]
    low, high = bootstrap_ci(differences, rng)
    paired = {"comparison": "GraphSAGE + MaskablePPO - Plain MaskablePPO", "metric": "average_waiting", "mean_difference": float(np.mean(differences)), "ci_95_low": low, "ci_95_high": high, "graph_better": high < 0}
    with open(PAIRED, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired))
        writer.writeheader(); writer.writerow(paired)
    with open(os.path.join(OUT, "summary_results.csv"), newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))
    fcfs = next(row for row in baseline_rows if row["model"] == "FCFS")
    graph_agents = [row for row in rows if row["model"] == "GraphSAGE + MaskablePPO"]
    versus_fcfs = []
    for metric in ("average_waiting", "priority_weighted_waiting"):
        differences = [float(row[metric]) - float(fcfs[metric]) for row in graph_agents]
        low, high = bootstrap_ci(differences, rng)
        versus_fcfs.append({"comparison": "GraphSAGE + MaskablePPO - FCFS", "metric": metric, "mean_difference": float(np.mean(differences)), "ci_95_low": low, "ci_95_high": high, "graph_better": high < 0})
    with open(VS_FCFS, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(versus_fcfs[0]))
        writer.writeheader(); writer.writerows(versus_fcfs)
    for row in output:
        print(f"{row['model']}: average waiting {row['average_waiting_mean']:.3f} [{row['average_waiting_ci_low']:.3f}, {row['average_waiting_ci_high']:.3f}]")
    print(f"Paired Graph-RL - Plain-RL: {paired['mean_difference']:.3f} [{paired['ci_95_low']:.3f}, {paired['ci_95_high']:.3f}]")
    for row in versus_fcfs:
        print(f"Graph-RL - FCFS {row['metric']}: {row['mean_difference']:.3f} [{row['ci_95_low']:.3f}, {row['ci_95_high']:.3f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("plain", "graph"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.kind and args.seed:
        evaluate_one(args.kind, args.seed)
    else:
        parser.error("use --kind KIND --seed SEED, or --summarize")


if __name__ == "__main__":
    main()
