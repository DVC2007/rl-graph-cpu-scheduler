from rl_env import CPUSchedulingEnv


env = CPUSchedulingEnv()

observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(3)

print("IDLE reward:", reward)


print("\nHidden process behavior:")

for process in env.processes:

    print(
        process["name"],
        "| Type:",
        process["type"],
        "| CPU bursts:",
        process["cpu_bursts"],
        "| I/O bursts:",
        process["io_bursts"]
    )


print("\nInitial state:")
print(observation)


print("\nTesting P1 repeatedly:")

for step in range(10):

    observation, reward, terminated, truncated, info = env.step(0)

    p1 = env.processes[0]

    print(
        f"\nTime {env.current_time}"
    )

    print(
        "P1 ready:",
        p1["ready"]
    )

    print(
        "P1 blocked time:",
        p1["blocked_time"]
    )

    print(
        "P1 last burst:",
        p1["last_burst"]
    )

    print(
        "P1 average burst:",
        p1["average_burst"]
    )

    print(
        "Reward:",
        reward
    )

    if terminated:
        break