from workload import processes

# Sort processes by burst time
sorted_processes = sorted(processes, key=lambda x: x[2])

current_time = 0
results = []

for name, arrival, burst in sorted_processes:
    start_time = current_time
    waiting_time = start_time - arrival

    finish_time = start_time + burst
    turnaround_time = finish_time - arrival

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

    current_time = finish_time


print("\n--- SJF RESULTS ---")

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


average_waiting = sum(r["waiting"] for r in results) / len(results)

average_turnaround = (
    sum(r["turnaround"] for r in results) / len(results)
)

print("\nAverage Waiting Time:", average_waiting)
print("Average Turnaround Time:", average_turnaround)