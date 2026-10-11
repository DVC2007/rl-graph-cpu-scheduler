# CPU Scheduling Lab Project History

## Purpose and evidence basis

This history reconstructs the project from the current source files, tests, checkpoints, README, benchmark CSV files, and the single available Git commit. Because the repository has only one visible commit dated 2026-09-19, the development sequence is described as logical phases rather than assigning unsupported dates.

The project began with a practical question: can reinforcement learning make CPU scheduling decisions that balance waiting time, completion time, process importance, and starvation? Classical schedulers use fixed rules. RL can learn a policy, but RL-based scheduling already exists, so the project later explored whether a dynamic process-interaction graph could add useful structure. The implementation remains a portable Python simulator, suitable for macOS and Apple Silicon, rather than a modification to the Linux kernel.

## Phase 1 Classical scheduling foundations

The earliest files model conventional scheduling with a shared tuple representation containing process name, arrival time, burst length, and later process type. The repository retains FCFS, SJF, and Round Robin implementations, a fixed example workload, a random workload generator, and scripts that compare schedulers on common workloads.

This phase was necessary for two reasons. First, it established the basic definitions of waiting and turnaround time. Second, it created reference behaviours against which learned policies could later be interpreted. `benchmark_schedulers.py` seeds Python's random generator and runs FCFS, SJF, and Round Robin on the same 100 generated workloads. SJF uses the real burst length and should therefore be treated as a privileged oracle baseline when compared with realistic RL policies.

## Phase 2 Process importance and strict priority

The workload generator added three process categories: SYSTEM, INTERACTIVE, and BACKGROUND. `type_priority_scheduler.py` introduced a strict ordering of 3, 2, and 1, while `analyze_by_type.py` measured waiting separately by category.

At this stage the project made process importance visible, but strict priority exposed a fairness problem: less important BACKGROUND work could wait excessively when more important work remained available. This motivated the later change from a hard ordering to soft weighted penalties and aging.

## Phase 3 First reinforcement learning prototype

The repository contains several checkpoints that record the progression of the RL work: `ppo_cpu_scheduler.zip`, `ppo_soft_priority_scheduler.zip`, and `ppo_randomized_soft_priority_scheduler.zip`. The first prototype used a Gymnasium environment and Stable-Baselines3 PPO. The scheduler state became an observation, selecting a process became an action, and waiting became part of the reward.

Early scripts test whether PPO learned a sensible process order. The fixed workload and fixed process positions were useful for proving that the training loop worked, but they were not sufficient evidence of general scheduling ability.

## Phase 4 Realism correction and randomized workloads

The project then addressed two sources of unfair advantage. Exact burst length is useful to the simulator, but a real scheduler does not know future CPU demand with certainty. Likewise, training repeatedly on fixed positions allows a policy to memorize process indices.

The environment was changed so that future CPU bursts, remaining future CPU work, future I/O duration, and future arrivals are not included in the policy observation. Hidden burst arrays remain inside the simulator only so it can execute the workload. Workloads and process types became randomized and reproducible through Gymnasium seeds.

The lesson from this phase was that a high score is not meaningful if the policy receives information that would be unavailable at decision time.

## Phase 5 Multi-burst CPU and I/O behaviour

`rl_env.py` extended each process from one CPU burst to alternating CPU and I/O bursts. Processes move between READY, BLOCKED, and FINISHED states. The environment also records observable history:

- waiting time
- process type
- CPU time used
- number of scheduling selections
- last completed CPU burst
- average completed CPU burst
- READY state
- blocked time

This history was necessary because two processes of the same type can behave differently. Without history, the agent has little evidence for distinguishing them once future bursts are hidden.

## Phase 6 Legal actions with IDLE and action masking

The next checkpoint family, including `maskable_ppo_cpu_scheduler.zip`, moved from ordinary PPO to MaskablePPO from `sb3-contrib`. The environment added an IDLE action and a Boolean action mask.

READY unfinished processes are legal. BLOCKED, FINISHED, and nonexistent processes are illegal. IDLE is legal only when no real process can run. Masking is more realistic than merely penalizing an invalid selection because an operating system cannot execute a blocked process; the choice should not enter the feasible action set at all.

## Phase 7 Soft priority and aging

The strict type hierarchy was replaced with soft waiting weights:

- BACKGROUND 1.0
- INTERACTIVE 1.5
- SYSTEM 2.0

Waiting by an important process therefore produces a larger penalty, but BACKGROUND work is not permanently forbidden from overtaking it. Once waiting exceeds the starvation threshold, an additional aging penalty grows with extra waiting. The threshold and aging factor became environment constructor parameters.

`maskable_ppo_aging_scheduler.zip` and `benchmark_rl_models.py` preserve the aging experiment. The earlier reported comparison showed a genuine trade-off: aging reduced maximum waiting and BACKGROUND waiting, while SYSTEM waiting and average waiting could worsen. This was reward-design evidence, not proof that one setting universally dominates.

## Phase 8 Plain RL benchmark discipline

Benchmarking evolved from demonstrations to repeated seeded episodes. The old `benchmark_rl_models.py` already ensured that aging and non-aging models received identical episode seeds. Later, `research_benchmark.py` corrected metric definitions and aggregation:

- mean episode maximum waiting is separated from the true global maximum
- starvation rate is aggregated across real processes
- process-type waiting is calculated from sums and counts
- an assertion verifies that weighted type averages agree with overall waiting
- Plain PPO and Graph-RL use identical seeds 10000 through 10499

This phase showed that metric definitions can change the interpretation of a result. Reproducibility and correct aggregation are part of the research method, not presentation details.

## Phase 9 Graph engineering and the first GCN

`process_graph.py` models each real process as a graph node. Its ten node features are built only from observable state and history. Directed edges are added for three relationships:

1. both processes are READY and competing for the CPU
2. both processes have the same type
3. both processes have similar observed average completed CPU bursts

Self-loops preserve each node's own information. The graph never uses hidden future burst or I/O values.

`gnn_model.py` records the first GCN experiment. It uses two `GCNConv` layers and produces a score per process. Dense contention graphs caused features to mix aggressively, producing nearly identical node scores. This is the practical over-smoothing problem that motivated the move to GraphSAGE-style aggregation.

## Phase 10 GraphSAGE integration with MaskablePPO

`graph_policy.py` implements a custom Stable-Baselines3 feature extractor. It expands the observable process type into one-hot features, builds dynamic relationships from visible state, computes neighbour means, concatenates each process's own features with its neighbours, and applies two learned layers.

The embeddings are fed into MaskablePPO. The graph therefore influences the policy; it is not a separate diagnostic whose largest score is chosen manually. The current implementation merges the three supported relationships into one adjacency relation for learning. The frontend displays edge families separately, but the policy does not yet use relation-specific parameters.

## Phase 11 Variable process-count upgrade

The first RL and graph checkpoints assumed exactly three processes. PPO requires fixed observation and action dimensions, which made direct variable-size input impossible.

The current solution defines `MAX_PROCESSES = 8`:

- the observation always has eight slots with eight observable values each
- unused slots are zero padded
- actions 0 through 7 select P1 through P8
- action 8 is IDLE
- unused slots are masked out
- the graph contains only actual processes
- GraphSAGE embeddings are mapped back into the corresponding fixed PPO slots

`test_variable_processes.py` checks 1, 3, 5, and 8 processes, padded observations, nine action-mask entries, graph node counts, IDLE legality, model rollout, and feature-extractor output. Training environments randomize active workload size from 3 to 8. New checkpoints were saved separately as `maskable_ppo_variable_scheduler.zip` and `graphsage_maskable_ppo_variable_scheduler.zip`, preserving the earlier three-process models.

## Phase 12 Current variable-size benchmark

The latest verified `results/summary_results.csv` contains a 500-workload comparison at aging factor 0.5.

| Metric | Plain MaskablePPO | GraphSAGE and MaskablePPO |
|---|---:|---:|
| Average waiting | 21.653 | 20.824 |
| Mean episode maximum waiting | 38.024 | 37.552 |
| Global maximum waiting | 79 | 83 |
| Completion time | 53.260 | 53.700 |
| Starvation rate | 65.390 percent | 62.934 percent |

Graph-RL improved average waiting, mean episode maximum waiting, and starvation rate. Plain PPO had slightly better completion time and a lower single global worst wait. The correct conclusion is that graph context changed the trade-off; it did not dominate every metric.

## Phase 13 Frontend and live explanation

`app.py` adds a Streamlit research dashboard. It loads the variable-size checkpoints and provides workload controls, Step mode, Run Episode, process cards, a CPU timeline, dynamic graph, READY and BLOCKED lists, action mask, decision context, live metrics, same-seed comparison, benchmark CSV viewing, and a researcher-only hidden-workload view.

The frontend was added because aggregate benchmark numbers do not explain scheduler behaviour. Step mode makes every transition inspectable: which action was legal, which process ran, which processes waited, when I/O caused blocking, and how the graph changed.

## Challenges and lessons learned

| Problem | Why it mattered | Fix | What we learned |
|---|---|---|---|
| Future burst information was visible | It gave RL an unrealistic oracle advantage | Keep bursts inside the simulator but remove them from observations | Realism depends on the information boundary |
| Fixed workloads and positions | The policy could memorize indices | Generate seeded random workloads | Generalization needs varied training cases |
| Strict type priority | BACKGROUND work could starve | Use soft type weights and aging | Importance and fairness must be balanced |
| BLOCKED and FINISHED actions | Penalizing impossible actions wastes learning and is unrealistic | Use MaskablePPO and legal-action masks | Feasibility belongs in the action space |
| Single CPU bursts | Processes did not resemble interactive workloads | Add alternating CPU and I/O bursts with state transitions | Scheduling quality depends on temporal behaviour |
| Hidden future removed | Same-type processes became difficult to distinguish | Add observable CPU and blocking history | Historical evidence can replace forbidden future knowledge |
| GCN node scores became similar | Dense graphs over-mixed process features | Use GraphSAGE-style self plus neighbour aggregation | Graph architecture must preserve node identity |
| Three-process model only | PPO dimensions were fixed | Eight padded slots, fixed actions, masks, and real-node graphs | Variable entities can be mapped into a fixed policy interface |
| Ambiguous maximum and type aggregation | Results could be described incorrectly | Separate mean episode maximum and global maximum; aggregate by counts | Metric definitions must be audited |
| Average performance versus fairness | One model did not win every metric | Report multiple paired metrics | Scheduling is a trade-off, not a single-score problem |

## What was changed or tuned

### Algorithmic design changes

- classical rules to PPO policy learning
- single burst to CPU and I/O state transitions
- ordinary PPO to action-masked PPO
- flat observation baseline to graph-aware feature extraction
- fixed three-process input to variable real-process graphs with fixed padded policy slots

### Reward and policy-design choices

- type waiting weights of 1.0, 1.5, and 2.0
- starvation threshold around 10
- aging factor, currently benchmarked at 0.5
- invalid-action masking

These are not all generic hyperparameters. Type weights, aging, masking, and observation visibility encode the scheduling problem itself.

### Model and training choices

- PPO and MaskablePPO
- GCN diagnostic followed by GraphSAGE-style aggregation
- hidden representation sizes and policy architecture
- training timesteps and random seeds
- random variable-size training workloads

## Current contribution

The project should be described conservatively as a study of dynamic process-interaction graphs combined with realistic observable history, soft process-type priority, CPU and I/O behaviour, action masking, and GraphSAGE-aware reinforcement learning under variable workloads with hidden future bursts. It should not be presented as the first GNN plus RL scheduler or as a completely new RL algorithm.

## Current limitations

- It is a simulator, not a real Linux kernel scheduler.
- Relationships are derived from simulated state rather than measured IPC or resource-sharing traces.
- All graph edge families currently share one aggregation mechanism.
- The starvation threshold needs stronger empirical justification.
- Publication-quality claims require more training seeds, confidence intervals, and statistical tests.
- Classical schedulers should be adapted to the same multi-burst CPU and I/O environment.
- The current fixed policy interface supports at most eight processes.

## Future work

- Evaluate unseen process counts and workload distributions.
- Run multiple training seeds and statistical significance tests.
- Perform edge ablations for contention, type, and history relations.
- Add relation-specific graph layers.
- Introduce justified wake-up, dependency, IPC, or resource-sharing edges only when the simulator or traces provide them.
- Evaluate real workload traces.
- Extend traditional baselines to the realistic CPU and I/O simulator.
- Explore a Linux `sched_ext` prototype if a suitable environment becomes available.

