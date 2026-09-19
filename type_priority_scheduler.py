PROCESS_PRIORITY = {
    "SYSTEM": 3,
    "INTERACTIVE": 2,
    "BACKGROUND": 1,
}


def run_type_priority(processes):
    remaining_processes = list(processes)

    current_time = 0
    results = []

    while remaining_processes:

        # Only processes that have already arrived
        ready_processes = [
            process
            for process in remaining_processes
            if process[1] <= current_time
        ]

        # If nobody has arrived yet,
        # jump to the next arrival
        if not ready_processes:
            current_time = min(
                process[1]
                for process in remaining_processes
            )
            continue

        # Choose the process with the HIGHEST type priority
        selected_process = max(
            ready_processes,
            key=lambda x: PROCESS_PRIORITY[x[3]]
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
        "name": "Type Priority",
        "average_waiting": average_waiting,
        "average_turnaround": average_turnaround,
        "results": results,
    }