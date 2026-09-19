def run_round_robin(processes, time_quantum=2):
    # Sort processes by arrival time
    processes = sorted(processes, key=lambda x: x[1])

    current_time = 0
    queue = []
    results = []

    # Convert processes into dictionaries
    all_processes = []

    for name, arrival, burst, process_type in processes:
        all_processes.append(
            {
                "name": name,
                "arrival": arrival,
                "burst": burst,
                "type": process_type,
                "remaining": burst,
                "finish": None,
            }
        )

    # Index tells us which process has not yet entered the queue
    index = 0

    while queue or index < len(all_processes):

        # Add all processes that have arrived by current_time
        while (
            index < len(all_processes)
            and all_processes[index]["arrival"] <= current_time
        ):
            queue.append(all_processes[index])
            index += 1

        # If nobody is ready, jump forward to next arrival
        if not queue:
            current_time = all_processes[index]["arrival"]
            continue

        # Take first process from queue
        process = queue.pop(0)

        run_time = min(time_quantum, process["remaining"])

        current_time += run_time
        process["remaining"] -= run_time

        # While this process was running,
        # new processes may have arrived
        while (
            index < len(all_processes)
            and all_processes[index]["arrival"] <= current_time
        ):
            queue.append(all_processes[index])
            index += 1

        # If process still has work left,
        # put it back at end of queue
        if process["remaining"] > 0:
            queue.append(process)

        else:
            process["finish"] = current_time

            turnaround = process["finish"] - process["arrival"]

            waiting = turnaround - process["burst"]

            results.append(
                {
                    "name": process["name"],
                    "arrival": process["arrival"],
                    "burst": process["burst"],
                    "type": process["type"],
                    "finish": process["finish"],
                    "waiting": waiting,
                    "turnaround": turnaround,
                }
            )

    average_waiting = (
        sum(r["waiting"] for r in results) / len(results)
    )

    average_turnaround = (
        sum(r["turnaround"] for r in results) / len(results)
    )

    return {
        "name": "Round Robin",
        "average_waiting": average_waiting,
        "average_turnaround": average_turnaround,
        "results": results,
    }