import random
from generate_workloads import generate_workload
from fcfs_scheduler import run_fcfs
from sjf_scheduler import run_sjf
from round_robin_scheduler import run_round_robin
random.seed(42)

NUM_TESTS = 100
NUM_PROCESSES = 5


fcfs_waiting = []
sjf_waiting = []
rr_waiting = []

fcfs_turnaround = []
sjf_turnaround = []
rr_turnaround = []


for i in range(NUM_TESTS):

    # Generate a NEW workload
    processes = generate_workload(NUM_PROCESSES)

    # Run all schedulers on the SAME workload
    fcfs = run_fcfs(processes)
    sjf = run_sjf(processes)
    rr = run_round_robin(processes, time_quantum=2)

    # Save waiting times
    fcfs_waiting.append(fcfs["average_waiting"])
    sjf_waiting.append(sjf["average_waiting"])
    rr_waiting.append(rr["average_waiting"])

    # Save turnaround times
    fcfs_turnaround.append(fcfs["average_turnaround"])
    sjf_turnaround.append(sjf["average_turnaround"])
    rr_turnaround.append(rr["average_turnaround"])


# Calculate overall averages

avg_fcfs_waiting = sum(fcfs_waiting) / NUM_TESTS
avg_sjf_waiting = sum(sjf_waiting) / NUM_TESTS
avg_rr_waiting = sum(rr_waiting) / NUM_TESTS

avg_fcfs_turnaround = sum(fcfs_turnaround) / NUM_TESTS
avg_sjf_turnaround = sum(sjf_turnaround) / NUM_TESTS
avg_rr_turnaround = sum(rr_turnaround) / NUM_TESTS


print("\n--- RESULTS AFTER 100 RANDOM WORKLOADS ---")

print(
    f"FCFS | "
    f"Average Waiting: {avg_fcfs_waiting:.2f} | "
    f"Average Turnaround: {avg_fcfs_turnaround:.2f}"
)

print(
    f"SJF | "
    f"Average Waiting: {avg_sjf_waiting:.2f} | "
    f"Average Turnaround: {avg_sjf_turnaround:.2f}"
)

print(
    f"Round Robin | "
    f"Average Waiting: {avg_rr_waiting:.2f} | "
    f"Average Turnaround: {avg_rr_turnaround:.2f}"
)