"""Train five reproducible Plain-RL and Graph-RL checkpoints for seed analysis."""
import argparse
import json
import os

from sb3_contrib import MaskablePPO

from graph_policy import GraphSAGEFeatureExtractor
from rl_env import CPUSchedulingEnv

ROOT = os.path.dirname(__file__)
OUT = os.path.join(ROOT, "results", "checkpoints")
AGING_FACTOR = 0.5


def make_model(kind, seed):
    env = CPUSchedulingEnv(num_processes=8, random_process_count=True, starvation_threshold=10, aging_factor=AGING_FACTOR)
    if kind == "plain":
        return MaskablePPO("MlpPolicy", env, seed=seed, verbose=0)
    policy_kwargs = {
        "features_extractor_class": GraphSAGEFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 48, "max_processes": 8},
        "net_arch": {"pi": [64], "vf": [64]},
    }
    return MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, seed=seed, verbose=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("plain", "graph", "both"), default="both")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    kinds = ("plain", "graph") if args.model == "both" else (args.model,)
    for kind in kinds:
        for seed in args.seeds:
            print(f"Training {kind} seed={seed} for {args.timesteps} timesteps")
            model = make_model(kind, seed)
            model.learn(total_timesteps=args.timesteps)
            model.save(os.path.join(OUT, f"{kind}_seed_{seed}"))
    with open(os.path.join(OUT, "training_config.json"), "w") as handle:
        json.dump({"aging_factor": AGING_FACTOR, "timesteps": args.timesteps, "seeds": args.seeds, "models": kinds}, handle, indent=2)


if __name__ == "__main__":
    main()
