from stable_baselines3 import PPO
from rl_env import CPUSchedulingEnv


env = CPUSchedulingEnv()

model = PPO.load("ppo_cpu_scheduler")


observation, info = env.reset()

print("\n--- RL SCHEDULER TEST ---")

finished = False

while not finished:

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    print("RL selected process index:", action)

    observation, reward, finished, truncated, info = env.step(action)

    print("New state:", observation)
    print("Reward:", reward)
    print()