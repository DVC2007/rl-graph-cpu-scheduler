from sb3_contrib import MaskablePPO
from rl_env import CPUSchedulingEnv


env = CPUSchedulingEnv()

model = MaskablePPO.load(
    "maskable_ppo_aging_scheduler"
)


for episode in range(5):

    observation, info = env.reset()

    finished = False

    print(
        f"\n--- TEST EPISODE {episode + 1} ---"
    )

    print("\nHidden process behavior:")

    for process in env.processes:
        print(
            process["name"],
            "| Type:",
            process["type"],
            "| CPU:",
            process["cpu_bursts"],
            "| IO:",
            process["io_bursts"]
        )

    print("\nRL decisions:")

    while not finished:

        action_mask = env.action_masks()

        action, _ = model.predict(
            observation,
            deterministic=True,
            action_masks=action_mask
        )

        action = int(action)

        if action == 3:

            print(
                f"Time {env.current_time}: IDLE"
            )

        else:

            selected_process = env.processes[action]

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