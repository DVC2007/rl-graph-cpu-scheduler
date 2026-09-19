def run_sjf(processes):
    # Make a copy so we do not modify the original workload
    remaining_processes = list(processes)

    current_time = 0
    results = []

    while remaining_processes:

        # Find processes that have already arrived
        ready_processes = [
            process
            for process in remaining_processes
            if process[1] <= current_time
        ]

        # If no process has arrived yet,
        # move time forward to the next arrival
        if not ready_processes:
            next_arrival = min(
                process[1]
                for process in remaining_processes
            )

            current_time = next_arrival

            continue

        # Choose the process with the smallest burst time
        selected_process = min(
            ready_processes,
            key=lambda x: x[2]
        )

        name, arrival, burst, process_type = selected_process

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

        # This process has finished,
        # so remove it from the waiting list
        remaining_processes.remove(selected_process)

    average_waiting = (
        sum(r["waiting"] for r in results)
        / len(results)
    )

    average_turnaround = (
        sum(r["turnaround"] for r in results)
        / len(results)
    )

    return {
        "name": "SJF",
        "average_waiting": average_waiting,
        "average_turnaround": average_turnaround,
        "results": results,
    }