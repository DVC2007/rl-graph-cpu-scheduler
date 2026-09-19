"""Reproducible baseline/Graph-RL benchmark with CSV and plots."""
import csv
import os
from collections import defaultdict
import numpy as np
from sb3_contrib import MaskablePPO
from rl_env import CPUSchedulingEnv

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
N = int(os.environ.get("RL_BENCHMARK_WORKLOADS", "500"))
THRESHOLD = 10
AGING_FACTOR = 0.5


def run(model, seed):
    env = CPUSchedulingEnv(num_processes=8, random_process_count=True,
                           starvation_threshold=THRESHOLD, aging_factor=AGING_FACTOR)
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        mask = env.action_masks()
        if model is None:
            ready = np.flatnonzero(mask[:-1])
            action = int(ready[0]) if len(ready) else env.num_processes
        else:
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            action = int(action)
        obs, _, done, _, _ = env.step(action)
    waits = [p["waiting"] for p in env.processes]
    by_type = defaultdict(list)
    for p in env.processes:
        by_type[p["type"]].append(p["waiting"])
    return {"average_waiting": float(np.mean(waits)),
            "episode_max_waiting": max(waits),
            "global_max_waiting": max(waits),
            "completion_time": env.current_time,
            "starvation_count": sum(w > THRESHOLD for w in waits),
            "process_count": len(waits),
            "SYSTEM_waiting_sum": sum(by_type["SYSTEM"]), "SYSTEM_count": len(by_type["SYSTEM"]),
            "INTERACTIVE_waiting_sum": sum(by_type["INTERACTIVE"]), "INTERACTIVE_count": len(by_type["INTERACTIVE"]),
            "BACKGROUND_waiting_sum": sum(by_type["BACKGROUND"]), "BACKGROUND_count": len(by_type["BACKGROUND"])}


def evaluate(name, model):
    rows = []
    for i in range(N):
        row = run(model, 10_000 + i)
        row.update(model=name, aging_factor=AGING_FACTOR, workload_seed=10_000 + i)
        rows.append(row)
    return rows


def main():
    models = []
    for path, name in [("maskable_ppo_variable_scheduler", "Plain MaskablePPO"),
                       ("graphsage_maskable_ppo_variable_scheduler", "GraphSAGE + MaskablePPO")]:
        if os.path.exists(os.path.join(os.path.dirname(__file__), path + ".zip")):
            models.append((name, MaskablePPO.load(os.path.join(os.path.dirname(__file__), path))))
    rows = []
    for name, model in models:
        rows.extend(evaluate(name, model))
    raw = os.path.join(OUT, "raw_results.csv")
    with open(raw, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary_rows = []
    for model, _ in models:
        values = [r for r in rows if r["model"] == model]
        total_processes = sum(r["process_count"] for r in values)
        item = {"model": model, "aging_factor": AGING_FACTOR}
        item["average_waiting"] = sum(r["average_waiting"] * r["process_count"] for r in values) / total_processes
        item["mean_episode_max_waiting"] = float(np.mean([r["episode_max_waiting"] for r in values]))
        item["global_max_waiting"] = max(r["global_max_waiting"] for r in values)
        item["completion_time"] = float(np.mean([r["completion_time"] for r in values]))
        item["starvation_rate"] = sum(r["starvation_count"] for r in values) / total_processes
        for typ in ("SYSTEM", "INTERACTIVE", "BACKGROUND"):
            count = sum(r[typ + "_count"] for r in values)
            item[typ + "_waiting"] = sum(r[typ + "_waiting_sum"] for r in values) / count if count else 0.0
        weighted = sum(item[t + "_waiting"] * sum(r[t + "_count"] for r in values)
                       for t in ("SYSTEM", "INTERACTIVE", "BACKGROUND")) / total_processes
        assert np.isclose(weighted, item["average_waiting"]), f"type-weighted waiting mismatch for {model}"
        summary_rows.append(item)
    with open(os.path.join(OUT, "summary_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0])); writer.writeheader(); writer.writerows(summary_rows)
    print(f"\nBenchmark: {N} identical seeded workloads; aging_factor={AGING_FACTOR}")
    print("model | average waiting | mean episode maximum | global maximum | completion time | starvation rate | SYSTEM | INTERACTIVE | BACKGROUND")
    for r in summary_rows:
        print(f"{r['model']} | {r['average_waiting']:.3f} | {r['mean_episode_max_waiting']:.3f} | {r['global_max_waiting']:.3f} | {r['completion_time']:.3f} | {r['starvation_rate']:.3%} | {r['SYSTEM_waiting']:.3f} | {r['INTERACTIVE_waiting']:.3f} | {r['BACKGROUND_waiting']:.3f}")
    print("Wrote", raw, "and summary_results.csv")


if __name__ == "__main__": main()
