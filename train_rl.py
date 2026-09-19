from sb3_contrib import MaskablePPO
from rl_env import CPUSchedulingEnv


env = CPUSchedulingEnv()


model = MaskablePPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=42
)


model.learn(
    total_timesteps=150000
)


model.save(
    "maskable_ppo_aging_scheduler"
)


print("\nTraining complete.")
print(
    "Model saved as: "
    "maskable_ppo_aging_scheduler"
)