from rl_env import CPUSchedulingEnv


env = CPUSchedulingEnv()

observation, info = env.reset()


print("\nHidden CPU / I/O behavior:")

for process in env.processes:
    print(
        process["name"],
        "| CPU:",
        process["cpu_bursts"],
        "| IO:",
        process["io_bursts"]
    )


print("\nInitial mask:")
print(env.action_masks())


# Run P1 repeatedly until it becomes blocked
while env.processes[0]["ready"]:

    observation, reward, terminated, truncated, info = env.step(0)

    if terminated:
        break


print("\nAfter P1 becomes blocked:")

print("P1 ready:", env.processes[0]["ready"])

print("Action mask:")
print(env.action_masks())