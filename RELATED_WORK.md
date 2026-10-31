# Related Work

## Deep RL for Resource Scheduling

**DeepRM** [Mao et al., HotNets 2016] introduced deep reinforcement learning to resource scheduling by representing cluster state as a 2D image of CPU and memory occupancy and training a REINFORCE policy to select jobs. While DeepRM demonstrated that RL can outperform heuristics such as SJF and Tetris, its image-based state representation is unscalable and encodes no inter-process relationships. Our work operates at OS process granularity — making per-timeslice dispatch decisions rather than cluster-level job allocations — and replaces the flat image with a dynamic graph that captures runtime process contention and behavioral similarity.

**RLScheduler** [Zhang et al., SC 2020] advanced RL scheduling for HPC batch workloads by switching to per-job feature vectors processed by a shared MLP and training with PPO. RLScheduler matches or exceeds SJF, WFP3, and UNICEP on six real HPC traces with 0.3 ms inference. However, it operates on minute-to-hour batch jobs rather than millisecond CPU time slices, uses a flat per-job vector that treats processes independently with no relational structure, has no action masking to enforce scheduling legality, and includes no priority or anti-starvation term in its reward. Our work addresses each of these gaps in the CPU scheduling domain.

## Graph Neural Networks for Scheduling

**Decima** [Mao et al., SIGCOMM 2019] is the landmark GNN + RL scheduler, representing Spark DAG stages as graph nodes and using a custom two-transform GNN with policy gradient to achieve 21% average job completion time improvement on a 25-node Spark cluster. Decima's graph is a static directed acyclic graph encoding data dependencies; it is built once per job arrival and does not change during execution. Our graph is fundamentally different: it is reconstructed every scheduler timeslice from observable process state (contention, type similarity, burst-history similarity), capturing dynamic runtime relationships that do not exist in static dependency graphs. Decima also reports sensitivity to workload distribution shift and includes no priority or fairness mechanism.

**L2D** [Zhang et al., NeurIPS 2020] applied GIN + PPO to manufacturing job-shop scheduling using a disjunctive graph encoding job-machine constraints, achieving a size-agnostic policy that generalizes to larger instances. **Geometric DAG RL** [Grinsztajn et al., 2020] extended this to HPC task DAGs using GCN + A2C. Both works confirm the value of graph-structured state for scheduling but remain in manufacturing or HPC domains with no process priority, no action masking, and no starvation protection.

A 2024 survey of GNN approaches to job-shop scheduling [arXiv 2406.14096] confirms that GCN, GIN, and GAT dominate the scheduling literature; **GraphSAGE** [Hamilton et al., NeurIPS 2017] — designed for inductive learning on dynamic graphs via neighborhood aggregation — has not been applied to any scheduling domain. We select GraphSAGE because the scheduler's graph is dynamic (rebuilt each timestep), inductive (must generalize to variable process counts), and amenable to the mean-aggregation that handles variable-degree contention neighborhoods.

**GNN-RL HPC** [Adimora et al., SC 2024] recently applied GAT + PPO to 377K real ALCF supercomputer jobs, citing GAT's inductiveness as essential for variable-size job queues. Our work applies the same inductive motivation to the finer granularity of CPU process scheduling and additionally incorporates action masking and priority-aware reward design absent from the HPC work.

## Action Masking in RL

**Huang & Ontañón** [FLAIRS 2022] established the theoretical and empirical foundation for action masking in policy gradient methods, proving that replacing invalid-action logits with −∞ before softmax yields valid policy gradients for the constrained policy. Compared to penalty-based alternatives, masking achieves a first reward 57× faster and scales gracefully as action space grows. We apply this via **MaskablePPO** [sb3-contrib] to enforce that BLOCKED and FINISHED processes cannot be selected and that IDLE is only taken when no process is ready.

**Lassoued et al.** [Expert Systems with Applications, 2026] — the most directly related work — applies MaskablePPO to dynamic job-shop scheduling under machine failures, masking infeasible job-machine transitions via Petri net guard functions. Their 100-run evaluation with 95% confidence intervals is the most rigorous statistical methodology found in the action-masking scheduling literature. Our work extends the MaskablePPO approach into the CPU scheduling domain, adds GraphSAGE-based graph state representation, introduces soft priority + aging reward design, and evaluates across 500 seeded workloads — exceeding their evaluation rigor.

## Priority and Fairness in Learned Schedulers

The majority of RL schedulers — DeepRM, Decima, RLScheduler, and L2D — optimize a single efficiency metric (average JCT or makespan) with no explicit process priority or anti-starvation fairness. The classical remedy for pure-priority starvation is aging [Dijkstra 1968], which gradually increases the effective priority of long-waiting processes. A 2025 GPU cluster scheduler [arXiv 2512.10980] applies aging as a hand-engineered heuristic scoring function. To our knowledge, no prior RL work formalizes aging as a **learned reward signal**: our reward combines a type-based priority weight with an aging penalty that grows super-linearly once waiting time exceeds a starvation threshold, enabling the policy to learn the priority–fairness tradeoff from experience rather than encoding it as a fixed rule.

## OS Kernel Scheduling and RL

**STUN** [Lee et al., Applied Sciences 2022] applies tabular Q-learning to tune Linux CFS global parameters (time quanta, scheduling policies), achieving 18.3% execution time speedup on face detection. Critically, STUN is a *parameter tuner*, not a per-process scheduler: it optimizes kernel configuration knobs rather than making per-timeslice dispatch decisions. TuneAgent [arXiv 2025] similarly uses LLM-guided RL for Linux kernel parameter tuning, gaining 5.6% over baselines. **No published work applies deep RL to per-timeslice, per-process CPU dispatch decisions in a scheduling simulator with type-differentiated processes, observable burst history, and anti-starvation reward design.** Our work fills this gap.

## Summary of Positioning

| Property | DeepRM | RLScheduler | Decima | L2D | Lassoued'26 | **Ours** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| CPU process scheduling | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Graph-structured state | ✗ | ✗ | ✓ | ✓ | ✗ | **✓** |
| GraphSAGE architecture | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Action masking (MaskablePPO) | ✗ | ✗ | partial | ✗ | ✓ | **✓** |
| Soft priority + aging reward | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Variable process count | ✗ | ✗ | ✗ | partial | ✗ | **✓** |
| Statistical CI evaluation (≥100 runs) | ✗ | ✗ | ✗ | ✗ | ✓ (100) | **✓ (500)** |

## References

- Mao et al. "Resource Management with Deep Reinforcement Learning." ACM HotNets, 2016.
- Zhang et al. "RLScheduler: An Automated HPC Batch Job Scheduler Using Reinforcement Learning." SC, 2020.
- Mao et al. "Learning Scheduling Algorithms for Data Processing Clusters." ACM SIGCOMM, 2019.
- Zhang et al. "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning." NeurIPS, 2020.
- Grinsztajn et al. "Geometric Deep Reinforcement Learning for Dynamic DAG Scheduling." arXiv:2011.04333, 2020.
- Adimora et al. "Deep Reinforcement Learning for Efficient HPC Job Scheduling." SC, 2024.
- Hamilton et al. "Inductive Representation Learning on Large Graphs." NeurIPS, 2017.
- Huang & Ontañón. "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms." FLAIRS, 2022.
- Lassoued et al. "Dynamic Job Shop Scheduling with Maskable PPO." Expert Systems with Applications, 2026.
- Lee et al. "STUN: Reinforcement-Learning-Based Optimization of Kernel Scheduler Parameters." Applied Sciences, 2022.
- Arpaci-Dusseau & Arpaci-Dusseau. "Operating Systems: Three Easy Pieces." Ch. 8 (MLFQ), 2018.
- Silberschatz et al. "Operating System Concepts." 10th ed., Ch. 5 (CPU burst distributions), 2018.
- Coffman & Kleinrock. "Feedback queueing models for time-shared systems." JACM, 1973.
