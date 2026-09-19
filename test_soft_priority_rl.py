from stable_baselines3 import PPO
from rl_env import CPUSchedulingEnv


env = CPUSchedulingEnv()

model = PPO.load("ppo_soft_priority_scheduler")

observation, info = env.reset()

finished = False

print("\n--- SOFT PRIORITY RL TEST ---")

while not finished:

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    selected_process = env.processes[int(action)]

    print(
        f"Time {env.current_time}: "
        f"RL chooses {selected_process['name']} "
        f"({selected_process['type']})"
    )

    observation, reward, finished, truncated, info = env.step(action)

    print("Reward:", reward)

    print("State:", observation)

    print()