def run_fcfs(processes):
    # FCFS = First Come First Served
    # Sort by arrival time
    ordered_processes = sorted(processes, key=lambda x: x[1])

    current_time = 0
    results = []

    for name, arrival, burst, process_type in ordered_processes:

        # If CPU is idle before this process arrives,
        # move time forward to its arrival
        if current_time < arrival:
            current_time = arrival

        start_time = current_time

        waiting_time = start_time - arrival

        finish_time = start_time + burst

        turnaround_time = finish_time - arrival

        results.append(
            {
                "name": name,
                "arrival": arrival,
                "burst": burst,
                "type": process_type,
                "start": start_time,
                "finish": finish_time,
                "waiting": waiting_time,
                "turnaround": turnaround_time,
            }
        )

        current_time = finish_time

    average_waiting = (
        sum(r["waiting"] for r in results) / len(results)
    )

    average_turnaround = (
        sum(r["turnaround"] for r in results) / len(results)
    )

    return {
        "name": "FCFS",
        "average_waiting": average_waiting,
        "average_turnaround": average_turnaround,
        "results": results,
    }