from stable_baselines3 import PPO
from rl_env import CPUSchedulingEnv


env = CPUSchedulingEnv()

model = PPO.load(
    "ppo_randomized_soft_priority_scheduler"
)


for episode in range(5):

    observation, info = env.reset()

    finished = False

    print(
        f"\n--- TEST EPISODE {episode + 1} ---"
    )

    print("Process types and hidden bursts:")

    for process in env.processes:
        print(
            process["name"],
            "| Type:",
            process["type"],
            "| Hidden burst:",
            process["total_burst"]
        )

    print("\nRL decisions:")

    while not finished:

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        selected_process = env.processes[
            int(action)
        ]

        print(
            f"Time {env.current_time}: "
            f"{selected_process['name']} "
            f"({selected_process['type']})"
        )

        observation, reward, finished, truncated, info = env.step(
            action
        )

    print("\nFinal waiting times:")

    for process in env.processes:
        print(
            process["name"],
            "|",
            process["type"],
            "| Waiting:",
            process["waiting"]
        )