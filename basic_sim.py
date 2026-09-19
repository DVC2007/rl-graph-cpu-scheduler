import simpy
from workload import processes

# This list will store the result of every process
results = []


def process(env, name, arrival, burst, cpu):
    # Process does not exist until its arrival time
    yield env.timeout(arrival)

    arrival_time = env.now

    print(f"{name} arrived at time {arrival_time}")

    # Ask for the CPU
    with cpu.request() as req:
        yield req

        # The process has now received the CPU
        start_time = env.now

        print(f"{name} started at time {start_time}")

        # Waiting time =
        # time CPU started process - time process arrived
        waiting_time = start_time - arrival_time

        # Process uses CPU for its burst time
        yield env.timeout(burst)

        finish_time = env.now

        print(f"{name} finished at time {finish_time}")

        # Turnaround time =
        # finish time - arrival time
        turnaround_time = finish_time - arrival_time

        results.append(
            {
                "name": name,
                "arrival": arrival,
                "burst": burst,
                "start": start_time,
                "finish": finish_time,
                "waiting": waiting_time,
                "turnaround": turnaround_time,
            }
        )


# ---------------------------------------------------
# Create our imaginary computer
# ---------------------------------------------------

env = simpy.Environment()

# capacity=1 means ONE CPU core
cpu = simpy.Resource(env, capacity=1)


# ---------------------------------------------------
# Our test workload
#
# Format:
# (Process Name, Arrival Time, Burst Time)
# ---------------------------------------------------




# Create each process inside our simulation
for name, arrival, burst in processes:
    env.process(
        process(
            env,
            name,
            arrival,
            burst,
            cpu
        )
    )


# Start the simulation
env.run()


# ---------------------------------------------------
# Print results
# ---------------------------------------------------

print("\n--- FCFS RESULTS ---")

for r in results:
    print(
        f"{r['name']} | "
        f"Arrival: {r['arrival']} | "
        f"Burst: {r['burst']} | "
        f"Start: {r['start']} | "
        f"Finish: {r['finish']} | "
        f"Waiting: {r['waiting']} | "
        f"Turnaround: {r['turnaround']}"
    )


# ---------------------------------------------------
# Calculate averages
# ---------------------------------------------------

average_waiting = sum(r["waiting"] for r in results) / len(results)

average_turnaround = (
    sum(r["turnaround"] for r in results) / len(results)
)


print("\nAverage Waiting Time:", average_waiting)

print(
    "Average Turnaround Time:",
    average_turnaround
)