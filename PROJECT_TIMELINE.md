# CPU Scheduling Lab Development Timeline

Exact milestone dates are unavailable because the repository exposes one initial commit dated 2026-09-19. The sequence below is reconstructed from retained scripts, checkpoints, tests, and current outputs.

| Phase | What we built | Why | Problem encountered | Improvement made | Result |
|---|---|---|---|---|---|
| 1 Classical foundations | Fixed workload, FCFS, SJF, Round Robin | Establish scheduling concepts and baselines | Separate scripts were difficult to compare | Shared workload tuples and comparison scripts | Waiting and turnaround became measurable |
| 2 Random benchmarking | Random workload generator and 100-test benchmark | Avoid conclusions from one example | Different workloads can make comparisons unfair | Seeded generation and same workload per scheduler | Reproducible classical comparison |
| 3 Process importance | SYSTEM, INTERACTIVE, BACKGROUND and Type Priority | Model different urgency levels | Strict priority can starve BACKGROUND work | Waiting analysis by process type | Fairness became an explicit objective |
| 4 First PPO | Gymnasium environment, PPO, early checkpoints | Learn scheduling instead of fixing one rule | Fixed workload and visible bursts allowed shortcuts | Randomized positions and workloads | Policy learned beyond one example |
| 5 Realism correction | Hidden future CPU and I/O information | Match what an OS can know at decision time | Removing burst knowledge made processes harder to distinguish | Observable current and historical features | Fairer RL problem definition |
| 6 CPU and I/O model | Multiple CPU bursts, I/O bursts, READY, BLOCKED, FINISHED | Represent realistic process behaviour | Single-burst tasks lacked state transitions | Burst history, CPU use, schedule count, blocked time | Temporal behaviour became observable without future leakage |
| 7 Legal actions | IDLE and MaskablePPO action masks | Prevent impossible process selections | BLOCKED and FINISHED actions were invalid | Mask illegal actions; allow IDLE only when required | Feasible decisions by construction |
| 8 Soft priority and aging | Weighted waiting penalties and anti-starvation logic | Respect importance without hard hierarchy | Strict priority harmed BACKGROUND fairness | Type weights plus growing aging penalty | Measurable priority and fairness trade-off |
| 9 Benchmark correction | Identical seeds and per-type metrics | Compare models fairly | Maximum-wait naming and type aggregation were ambiguous | Separate mean episode max and global max; aggregate sums and counts | Auditable metrics and CSV output |
| 10 Process graph | Process nodes, observable features, contention/type/history edges | Represent relationships absent from a flat vector | Dense graph mixing produced similar GCN scores | Retain self information and change graph architecture | Scheduling context became learnable |
| 11 GraphSAGE RL | Custom GraphSAGE-style feature extractor inside MaskablePPO | Ensure graph changes policy decisions | Standalone GNN scoring would be decorative | Feed graph embeddings into PPO policy | End-to-end Graph-RL checkpoint |
| 12 Variable processes | Eight fixed slots, nine actions, padding and masks | Move beyond the three-process limitation | PPO requires fixed dimensions | Real-node graphs mapped into fixed policy slots | Support for 1–8 processes; random 3–8 training |
| 13 Variable benchmark | 500 paired seeded workloads | Measure graph benefit under changing workload size | Graph-RL did not win every metric | Report performance and fairness together | Graph-RL improved average wait and starvation, but not global worst wait or completion |
| 14 Visualization | Streamlit dashboard and Step mode | Make learned behaviour explainable during a demo | Aggregate CSV results hide individual decisions | Timeline, cards, graph, queues, mask, comparison, benchmark viewer | Professor-friendly live demonstration |

