from sb3_contrib import MaskablePPO
from rl_env import CPUSchedulingEnv

env = CPUSchedulingEnv(num_processes=8, random_process_count=True, starvation_threshold=10, aging_factor=0.5)
model = MaskablePPO("MlpPolicy", env, seed=42, verbose=1)
model.learn(total_timesteps=150_000)
model.save("maskable_ppo_variable_scheduler")
