from workload import processes

# Time quantum means:
# how long one process is allowed to run before giving CPU to another process
TIME_QUANTUM = 2

# We assume all processes arrive at time 0 for now
queue = []

for name, arrival, burst in processes:
    queue.append(
        {
            "name": name,
            "arrival": arrival,
            "burst": burst,
            "remaining": burst,
            "finish": None,
        }
    )

current_time = 0
results = []

print("\n--- ROUND ROBIN EXECUTION ---")

while queue:
    process = queue.pop(0)

    # Process runs for either:
    # the time quantum OR whatever remaining time is left
    run_time = min(TIME_QUANTUM, process["remaining"])

    print(
        f"{process['name']} runs from "
        f"time {current_time} to {current_time + run_time}"
    )

    current_time += run_time
    process["remaining"] -= run_time

    # If process still has work left,
    # put it back at the end of the queue
    if process["remaining"] > 0:
        queue.append(process)

    # Otherwise it has finished
    else:
        process["finish"] = current_time

        turnaround = process["finish"] - process["arrival"]

        waiting = turnaround - process["burst"]

        results.append(
            {
                "name": process["name"],
                "arrival": process["arrival"],
                "burst": process["burst"],
                "finish": process["finish"],
                "waiting": waiting,
                "turnaround": turnaround,
            }
        )


print("\n--- ROUND ROBIN RESULTS ---")

for r in results:
    print(
        f"{r['name']} | "
        f"Arrival: {r['arrival']} | "
        f"Burst: {r['burst']} | "
        f"Finish: {r['finish']} | "
        f"Waiting: {r['waiting']} | "
        f"Turnaround: {r['turnaround']}"
    )


average_waiting = sum(r["waiting"] for r in results) / len(results)

average_turnaround = (
    sum(r["turnaround"] for r in results) / len(results)
)

print("\nAverage Waiting Time:", average_waiting)
print("Average Turnaround Time:", average_turnaround)