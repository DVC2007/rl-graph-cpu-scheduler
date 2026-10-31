"""Three-tier contribution analysis: Classical → RL → RL+Graph.

For each workload type, isolates:
  Tier 1: Classical best  (no ML, no oracle)
  Tier 2: RL only         (MaskablePPO, flat MLP, no graph)
  Tier 3: RL + GraphSAGE  (MaskablePPO + GraphSAGE feature extractor)
  Reference: SRTF oracle  (uses hidden future bursts)

Runs paired episode comparisons so the delta between tiers is measured on
identical workloads (same seeds), giving a true within-episode significance test.
"""
import csv
import os
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO

from rl_env import CPUSchedulingEnv, PROCESS_WAIT_WEIGHT
from scheduler_baselines import CFSPolicy, FCFSPolicy, SRTFPolicy
from realistic_workload import generate_workload

ROOT = os.path.dirname(__file__)
OUT  = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

N         = int(os.environ.get("RL_BENCHMARK_WORKLOADS", "500"))
THRESHOLD = 10
AGING     = 0.5
BOOTSTRAP = 5_000


class PPOPolicy:
    def __init__(self, model):
        self.model = model
    def act(self, env):
        a, _ = self.model.predict(
            env._get_observation(), deterministic=True, action_masks=env.action_masks()
        )
        return int(a)


def run_episode(policy, seed, workload):
    if workload is None:
        env = CPUSchedulingEnv(num_processes=8, random_process_count=True,
                               starvation_threshold=THRESHOLD, aging_factor=AGING)
        env.reset(seed=seed)
    else:
        env = CPUSchedulingEnv(num_processes=len(workload), starvation_threshold=THRESHOLD,
                               aging_factor=AGING)
        env.reset(seed=seed, options={"workload": workload})
    while not all(p["finished"] for p in env.processes):
        env.step(policy.act(env))
    waits = [p["waiting"] for p in env.processes]
    return {
        "avg_wait":     float(np.mean(waits)),
        "starvation":   float(sum(w > THRESHOLD for w in waits) / len(waits)),
        "pwait":        float(np.mean([PROCESS_WAIT_WEIGHT[p["type"]] * p["waiting"]
                                       for p in env.processes])),
        "max_wait":     float(max(waits)),
    }


def mean_ci(arr, rng):
    boot = np.mean(rng.choice(arr, size=(BOOTSTRAP, len(arr)), replace=True), axis=1)
    return float(np.mean(arr)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_test(a, b, rng, label_a, label_b):
    """Bootstrap paired difference b - a (negative = b is better)."""
    diff = np.asarray(b) - np.asarray(a)
    boot = np.mean(rng.choice(diff, size=(BOOTSTRAP, len(diff)), replace=True), axis=1)
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    sig = "✓ sig." if lo > 0 or hi < 0 else "✗ n.s."
    print(f"    Δ({label_b} − {label_a}): {np.mean(diff):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {sig}")
    return np.mean(diff), lo, hi


def main():
    # Load RL models
    rl_plain = MaskablePPO.load(os.path.join(ROOT, "maskable_ppo_variable_scheduler"))
    rl_graph = MaskablePPO.load(os.path.join(ROOT, "graphsage_maskable_ppo_variable_scheduler"))
    print("Models loaded.\n")

    rng = np.random.default_rng(2026)
    summary_rows = []

    for wl_name, base_seed, wl_factory in [
        ("uniform",   10_000, lambda s: None),
        ("realistic", 20_000, lambda s: generate_workload(s, num_processes=int(np.random.default_rng(s).integers(2,9)))),
    ]:
        print(f"{'═'*70}")
        print(f"  WORKLOAD TYPE: {wl_name.upper()}   (n={N} paired episodes)")
        print(f"{'═'*70}")

        workloads = [wl_factory(base_seed + i) for i in range(N)]

        # Pick the best classical (CFS for realistic, FCFS for uniform based on ablation)
        best_classical_name = "CFS" if wl_name == "realistic" else "FCFS"
        classical_policy    = CFSPolicy if wl_name == "realistic" else FCFSPolicy

        tiers = {
            best_classical_name: classical_policy,
            "RL only (MaskablePPO)":    lambda: PPOPolicy(rl_plain),
            "RL + GraphSAGE":           lambda: PPOPolicy(rl_graph),
            "SRTF (oracle)":            SRTFPolicy,
        }

        results = {name: [] for name in tiers}
        for i in range(N):
            seed = base_seed + i
            wl   = workloads[i]
            for name, factory in tiers.items():
                results[name].append(run_episode(factory(), seed, wl))

        # Per-tier summary
        print(f"\n  {'Tier':<28} {'Avg Wait':>10}  {'95% CI':>19}  {'Starvation':>12}  {'P-Wtd':>8}")
        print(f"  {'─'*28} {'─'*10}  {'─'*19}  {'─'*12}  {'─'*8}")
        tier_means = {}
        for name, rows in results.items():
            aw  = np.array([r["avg_wait"] for r in rows])
            st  = np.array([r["starvation"] for r in rows])
            pw  = np.array([r["pwait"] for r in rows])
            mu, lo, hi = mean_ci(aw, rng)
            tier_means[name] = {"aw": aw, "st": st, "pw": pw}
            tag = " †" if "oracle" in name else ""
            print(f"  {name+tag:<28} {mu:>10.2f}  [{lo:.2f}, {hi:.2f}]  "
                  f"{np.mean(st):>12.1%}  {np.mean(pw):>8.2f}")

        # Contribution decomposition
        print(f"\n  ── Average Wait: Paired Contribution Decomposition ──")
        cl_aw = tier_means[best_classical_name]["aw"]
        rl_aw = tier_means["RL only (MaskablePPO)"]["aw"]
        gr_aw = tier_means["RL + GraphSAGE"]["aw"]
        or_aw = tier_means["SRTF (oracle)"]["aw"]

        print(f"  Step 1 — RL contribution  (Classical → RL only):")
        d1, lo1, hi1 = paired_test(cl_aw, rl_aw, rng, best_classical_name, "RL only")

        print(f"  Step 2 — Graph contribution  (RL only → RL + GraphSAGE):")
        d2, lo2, hi2 = paired_test(rl_aw, gr_aw, rng, "RL only", "RL+Graph")

        print(f"  Step 3 — Remaining oracle gap  (RL+Graph → SRTF):")
        d3, lo3, hi3 = paired_test(gr_aw, or_aw, rng, "RL+Graph", "SRTF oracle")

        total_gap = np.mean(cl_aw) - np.mean(or_aw)
        rl_frac  = abs(d1) / total_gap * 100
        gr_frac  = abs(d2) / total_gap * 100
        print(f"\n  Total gap (classical → oracle): {total_gap:.2f} units")
        print(f"  RL closes: {abs(d1):.2f} units  ({rl_frac:.0f}% of gap)")
        print(f"  Graph closes: {abs(d2):.2f} units  ({gr_frac:.0f}% of gap)")
        print(f"  Remaining: {abs(d3):.2f} units  ({(abs(d3)/total_gap*100):.0f}% of gap)")

        print(f"\n  ── Starvation: Paired Contribution Decomposition ──")
        cl_st = tier_means[best_classical_name]["st"]
        rl_st = tier_means["RL only (MaskablePPO)"]["st"]
        gr_st = tier_means["RL + GraphSAGE"]["st"]
        print(f"  Classical ({best_classical_name}): {np.mean(cl_st):.1%}")
        print(f"  RL only:        {np.mean(rl_st):.1%}  (Δ = {np.mean(rl_st)-np.mean(cl_st):+.1%})")
        print(f"  RL + GraphSAGE: {np.mean(gr_st):.1%}  (Δ = {np.mean(gr_st)-np.mean(rl_st):+.1%})")
        paired_test(cl_st, rl_st, rng, best_classical_name, "RL only starvation")
        paired_test(rl_st, gr_st, rng, "RL only", "RL+Graph starvation")

        # Save
        for name, rows in results.items():
            aw = np.array([r["avg_wait"] for r in rows])
            st = np.array([r["starvation"] for r in rows])
            mu, lo, hi = mean_ci(aw, rng)
            summary_rows.append({
                "workload": wl_name, "tier": name,
                "avg_wait_mean": round(mu,3), "aw_ci_low": round(lo,3), "aw_ci_high": round(hi,3),
                "starvation_mean": round(float(np.mean(st)),4),
                "pwait_mean": round(float(np.mean([r["pwait"] for r in rows])),3),
            })

    with open(os.path.join(OUT, "contribution_analysis.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader(); writer.writerows(summary_rows)
    print(f"\n\nSaved: results/contribution_analysis.csv")
    print("† SRTF uses hidden future burst lengths — oracle lower bound only.")


if __name__ == "__main__":
    main()
