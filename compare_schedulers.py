from generate_workloads import generate_workload
from fcfs_scheduler import run_fcfs
from sjf_scheduler import run_sjf
from round_robin_scheduler import run_round_robin
from type_priority_scheduler import run_type_priority

# Generate one random workload with 5 processes
processes = generate_workload(5)

print("\n--- GENERATED WORKLOAD ---")

for process in processes:
    print(process)


# Run all schedulers on the SAME workload
fcfs_result = run_fcfs(processes)
sjf_result = run_sjf(processes)
rr_result = run_round_robin(processes, time_quantum=2)
priority_result = run_type_priority(processes)

print("\n--- SCHEDULER COMPARISON ---")

for result in [
    fcfs_result,
    sjf_result,
    rr_result,
    priority_result
]:
    print(
        f"{result['name']} | "
        f"Average Waiting: {result['average_waiting']:.2f} | "
        f"Average Turnaround: {result['average_turnaround']:.2f}"
    )