from sb3_contrib import MaskablePPO
from rl_env import CPUSchedulingEnv
from graph_policy import GraphSAGEFeatureExtractor


env = CPUSchedulingEnv(num_processes=8, random_process_count=True, starvation_threshold=10, aging_factor=0.5)
policy_kwargs = {
    "features_extractor_class": GraphSAGEFeatureExtractor,
    "features_extractor_kwargs": {"features_dim": 48, "max_processes": 8},
    "net_arch": {"pi": [64], "vf": [64]},
}
model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, seed=42, verbose=1)
model.learn(total_timesteps=150_000)
model.save("graphsage_maskable_ppo_variable_scheduler")
