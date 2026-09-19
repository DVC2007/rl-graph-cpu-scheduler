from generate_workloads import generate_workload
from type_priority_scheduler import run_type_priority


processes = generate_workload(20)

result = run_type_priority(processes)

waiting_by_type = {
    "SYSTEM": [],
    "INTERACTIVE": [],
    "BACKGROUND": []
}


for r in result["results"]:
    waiting_by_type[r["type"]].append(r["waiting"])


print("\n--- WAITING TIME BY PROCESS TYPE ---")

for process_type, waits in waiting_by_type.items():

    if waits:
        average_wait = sum(waits) / len(waits)
        maximum_wait = max(waits)

        print(
            f"{process_type} | "
            f"Average Waiting: {average_wait:.2f} | "
            f"Maximum Waiting: {maximum_wait:.2f}"
        )

    else:
        print(
            f"{process_type} | "
            f"No processes generated"
        )