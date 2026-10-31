"""SRTF gap benchmark: measures how much each model closes the oracle gap.

Four-tier comparison per workload type:
  Classical best → RL (flat) → RL+GraphSAGE → RL+Attention+SRPT-features → SRTF oracle

All tiers run on identical seeded workloads (paired).
"""
import os, csv
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO

from rl_env import CPUSchedulingEnv, PROCESS_WAIT_WEIGHT, IDLE_ACTION
from srpt_approx_env import SRTFApproxEnv
from scheduler_baselines import CFSPolicy, FCFSPolicy, SRTFPolicy
from realistic_workload import generate_workload

ROOT = os.path.dirname(__file__)
OUT  = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
N         = int(os.environ.get("RL_BENCHMARK_WORKLOADS", "500"))
THRESHOLD = 10
AGING     = 0.5
BOOTSTRAP = 5_000


class BasePPOPolicy:
    def __init__(self, model):
        self.model = model
    def act(self, env):
        a, _ = self.model.predict(env._get_observation(), deterministic=True,
                                  action_masks=env.action_masks())
        return int(a)


def run_base(policy, seed, workload):
    if workload is None:
        env = CPUSchedulingEnv(num_processes=8, random_process_count=True,
                               starvation_threshold=THRESHOLD, aging_factor=AGING)
        env.reset(seed=seed)
    else:
        env = CPUSchedulingEnv(num_processes=len(workload),
                               starvation_threshold=THRESHOLD, aging_factor=AGING)
        env.reset(seed=seed, options={"workload": workload})
    while not all(p["finished"] for p in env.processes):
        env.step(policy.act(env))
    waits = [p["waiting"] for p in env.processes]
    return np.mean(waits), np.mean([w > THRESHOLD for w in waits])


def run_srpt_env(model, seed, workload):
    """Run a model trained on SRTFApproxEnv (88-dim obs)."""
    if workload is None:
        env = SRTFApproxEnv(num_processes=8, random_process_count=True,
                            starvation_threshold=THRESHOLD, aging_factor=AGING)
        env.reset(seed=seed)
    else:
        env = SRTFApproxEnv(num_processes=len(workload),
                            starvation_threshold=THRESHOLD, aging_factor=AGING)
        env.reset(seed=seed, options={"workload": workload})
        for p in env.processes:
            p["ewma_burst"] = 0.0; p["burst_std"] = 0.0
    while not all(p["finished"] for p in env.processes):
        a, _ = model.predict(env._get_observation(), deterministic=True,
                             action_masks=env.action_masks())
        env.step(int(a))
    waits = [p["waiting"] for p in env.processes]
    return np.mean(waits), np.mean([w > THRESHOLD for w in waits])


def mean_ci(arr, rng):
    boot = np.mean(rng.choice(arr, size=(BOOTSTRAP, len(arr)), replace=True), axis=1)
    return float(np.mean(arr)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_delta_ci(a, b, rng):
    diff = np.asarray(b) - np.asarray(a)
    boot = np.mean(rng.choice(diff, size=(BOOTSTRAP, len(diff)), replace=True), axis=1)
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    sig = "✓" if lo > 0 or hi < 0 else "✗"
    return float(np.mean(diff)), lo, hi, sig


def main():
    rl_flat  = MaskablePPO.load(os.path.join(ROOT, "maskable_ppo_variable_scheduler"))
    rl_graph = MaskablePPO.load(os.path.join(ROOT, "graphsage_maskable_ppo_variable_scheduler"))
    rl_attn  = MaskablePPO.load(os.path.join(ROOT, "attention_graph_srpt_scheduler"))
    print("Models loaded.\n")
    rng = np.random.default_rng(2026)

    all_rows = []

    for wl_name, base_seed, wl_factory, best_classical_cls in [
        ("uniform",   10_000, lambda s: None,                  FCFSPolicy),
        ("realistic", 20_000, lambda s: generate_workload(s,
            num_processes=int(np.random.default_rng(s).integers(2,9))), CFSPolicy),
    ]:
        workloads = [wl_factory(base_seed + i) for i in range(N)]

        tiers = {}
        # Classical
        tiers["Classical (best)"] = [
            run_base(best_classical_cls(), base_seed + i, workloads[i]) for i in range(N)]
        # Oracle
        tiers["SRTF oracle"] = [
            run_base(SRTFPolicy(), base_seed + i, workloads[i]) for i in range(N)]
        # RL flat
        print(f"[{wl_name}] RL flat...")
        tiers["RL (flat MLP)"] = [
            run_base(BasePPOPolicy(rl_flat), base_seed + i, workloads[i]) for i in range(N)]
        # RL + GraphSAGE
        print(f"[{wl_name}] RL + GraphSAGE...")
        tiers["RL + GraphSAGE"] = [
            run_base(BasePPOPolicy(rl_graph), base_seed + i, workloads[i]) for i in range(N)]
        # RL + Attention + SRPT features
        print(f"[{wl_name}] RL + Attention + SRPT...")
        tiers["RL + Attn + SRPT"] = [
            run_srpt_env(rl_attn, base_seed + i, workloads[i]) for i in range(N)]

        aw  = {k: np.array([r[0] for r in v]) for k, v in tiers.items()}
        st  = {k: np.array([r[1] for r in v]) for k, v in tiers.items()}

        classical_aw = aw["Classical (best)"]
        oracle_aw    = aw["SRTF oracle"]
        total_gap    = float(np.mean(classical_aw) - np.mean(oracle_aw))

        print(f"\n{'═'*78}")
        print(f"  WORKLOAD: {wl_name.upper()}   (n={N})   Total classical→oracle gap: {total_gap:.2f} units")
        print(f"{'═'*78}")
        print(f"  {'Model':<24} {'Avg Wait':>10}  {'95% CI':>18}  {'Starvation':>11}  {'% gap closed':>13}")
        print(f"  {'─'*24} {'─'*10}  {'─'*18}  {'─'*11}  {'─'*13}")

        order = ["Classical (best)", "RL (flat MLP)", "RL + GraphSAGE",
                 "RL + Attn + SRPT", "SRTF oracle"]
        for name in order:
            mu, lo, hi = mean_ci(aw[name], rng)
            gap_closed = (np.mean(classical_aw) - mu) / total_gap * 100 if total_gap > 0 else 0
            oracle_tag = " †" if "oracle" in name else "  "
            print(f"  {name+oracle_tag:<24} {mu:>10.2f}  [{lo:.2f}, {hi:.2f}]  "
                  f"{np.mean(st[name]):>11.1%}  {gap_closed:>12.1f}%")
            all_rows.append({"workload": wl_name, "model": name,
                             "avg_wait": round(mu,3), "ci_low": round(lo,3), "ci_high": round(hi,3),
                             "starvation": round(float(np.mean(st[name])),4),
                             "gap_closed_pct": round(gap_closed,1)})

        # Paired step-by-step decomposition
        print(f"\n  ── Paired contribution per step ──")
        steps = [
            ("Classical → RL flat",     aw["Classical (best)"], aw["RL (flat MLP)"]),
            ("RL flat → RL+GraphSAGE",  aw["RL (flat MLP)"],    aw["RL + GraphSAGE"]),
            ("RL+Graph → RL+Attn+SRPT", aw["RL + GraphSAGE"],   aw["RL + Attn + SRPT"]),
            ("RL+Attn → SRTF oracle",   aw["RL + Attn + SRPT"], aw["SRTF oracle"]),
        ]
        for label, a_arr, b_arr in steps:
            d, lo, hi, sig = paired_delta_ci(a_arr, b_arr, rng)
            frac = abs(d) / total_gap * 100
            print(f"    {label:<34}  Δ={d:+.3f}  [{lo:+.3f},{hi:+.3f}]  {sig}  ({frac:.0f}% of gap)")

    with open(os.path.join(OUT, "srtf_gap_benchmark.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader(); writer.writerows(all_rows)

    print(f"\n† SRTF uses hidden future burst lengths — oracle lower bound, not a fair baseline.")
    print(f"Saved: results/srtf_gap_benchmark.csv")


if __name__ == "__main__":
    main()
