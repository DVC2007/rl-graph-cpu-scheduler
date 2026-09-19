import random


PROCESS_TYPES = [
    "SYSTEM",
    "INTERACTIVE",
    "BACKGROUND"
]


def generate_workload(num_processes=5):
    processes = []

    for i in range(num_processes):
        name = f"P{i+1}"

        arrival = random.randint(0, 10)

        burst = random.randint(1, 10)

        process_type = random.choice(PROCESS_TYPES)

        processes.append(
            (
                name,
                arrival,
                burst,
                process_type
            )
        )

    return processes


if __name__ == "__main__":
    workload = generate_workload(5)

    print("\nGenerated Workload:")

    for process in workload:
        print(process)