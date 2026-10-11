# CPU Scheduling Lab Professor Presentation Notes

## Five to seven minute script

### Speaker 1 Introduction

Our project began with the CPU scheduling problem: when several processes want one CPU core, which process should run next? We first implemented FCFS, Shortest Job First, and Round Robin. These baselines were important because they gave us correct definitions of waiting and turnaround time and showed what fixed scheduling rules look like.

We then added random workloads and common seeds. This was necessary because comparing schedulers on different workloads would not be fair.

### Speaker 2 Priority and first RL model

Next, we introduced SYSTEM, INTERACTIVE, and BACKGROUND process types. A strict Type Priority scheduler respected importance, but it showed a weakness: BACKGROUND processes could wait too long. That made starvation and fairness part of the research problem.

We then built a Gymnasium environment and trained PPO. At first, the setup was simple and used fixed workloads. PPO could learn a sensible order, but at this stage we noticed two unfair shortcuts. The policy could memorize process positions, and exact burst information told it more than a realistic operating system scheduler would know.

### Speaker 1 Realism correction

We changed the environment so future CPU bursts, remaining future work, and future I/O durations stayed hidden from the policy. The simulator still needs these values internally, but they are not part of the observation.

Removing future information created another problem: similar processes became difficult to distinguish. We therefore added realistic observable history, such as CPU time already used, number of times scheduled, last completed burst, average completed burst, and blocked time.

We also introduced alternating CPU and I/O bursts. Processes now move between READY, BLOCKED, and FINISHED states instead of behaving like one uninterrupted job.

### Speaker 2 Action masking, priority, and aging

A blocked process cannot physically use the CPU. Penalizing the model after it selects one is less realistic than preventing the action. We therefore moved to MaskablePPO. READY processes are legal, BLOCKED and FINISHED processes are masked, and IDLE is legal only when nobody can run.

We replaced strict priority with soft waiting weights: 2.0 for SYSTEM, 1.5 for INTERACTIVE, and 1.0 for BACKGROUND. Aging increases the penalty when a process waits beyond the threshold. This lets importance matter without permanently preventing an old BACKGROUND process from running.

### Speaker 1 Why a graph

Plain PPO sees a fixed vector, but scheduling is relational. Two READY processes compete for the CPU. Two processes may share a type or show similar historical CPU behaviour. We therefore made every process a graph node and connected nodes using those three relationships.

Our first GCN experiment produced nearly identical node scores because dense graphs mixed process features too strongly. We changed to GraphSAGE-style aggregation, which combines neighbour information with each process's own features and better preserves its identity.

The GraphSAGE embeddings are passed into MaskablePPO, so the graph affects the scheduling action. It is not only a visual diagram.

### Speaker 2 Variable workloads

The original model supported only three processes because PPO requires fixed input and output sizes. We introduced eight fixed process slots and one IDLE action. Unused observation slots contain zeros and unused actions are masked. The graph still contains only the real processes, and its embeddings are mapped back into the fixed PPO slots.

We tested 1, 3, 5, and 8 processes and trained on workloads that randomly vary from 3 to 8 processes.

### Speaker 1 Results

We evaluated both models on the same 500 seeded variable-size workloads. Plain PPO achieved average waiting of 21.653, while Graph-RL achieved 20.824. Mean episode maximum waiting improved from 38.024 to 37.552, and starvation rate improved from 65.390 percent to 62.934 percent.

However, Plain PPO had slightly better completion time, 53.260 compared with 53.700, and a better single global worst wait, 79 compared with 83. The result therefore shows a trade-off. Graph-RL improved several average and fairness measures, but did not dominate every metric.

### Speaker 2 Frontend and conclusion

Finally, we added a Streamlit dashboard because benchmark tables do not show why an episode behaves differently. Step mode displays the selected action, legal mask, READY and BLOCKED processes, CPU timeline, live metrics, and changing graph after every time unit. It can also compare Plain PPO and Graph-RL on the same seed and process count.

Our contribution is not a claim that we invented the first graph RL scheduler. It is an integrated study of realistic observable history, CPU and I/O state transitions, soft process priority, aging, action masking, dynamic process relationships, GraphSAGE-aware RL, and variable workloads without future-burst leakage.

The next work is stronger classical baselines in the same realistic environment, multiple training seeds, statistical significance tests, graph-edge ablations, real workload traces, and eventually a Linux `sched_ext` prototype if suitable access becomes available.

