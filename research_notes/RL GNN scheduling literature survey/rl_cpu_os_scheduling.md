# RL for CPU/OS Process Scheduling and Deep RL for OS Scheduling

## What RL algorithms have been applied to CPU/OS process scheduling?

### Takeaway
Most RL work on CPU/OS scheduling uses PPO, Q-learning, A2C, or policy gradient methods, but the majority operate on flat feature-vector state representations and none combine GNNs with MaskablePPO for per-process scheduling with priority aging.

### Cited Findings

- **RLScheduler (Zhang et al., 2019)**: Uses PPO with an actor-critic framework. Schedules HPC batch jobs (not OS process scheduling). State = per-job feature vectors (ID, user ID, submission time, requested processors, runtime estimate, memory) padded to MAX_OBSV_SIZE=128. Key contribution: kernel-based DNN applies 3-layer MLP to each job individually. Inference cost: 0.3ms for 128 jobs. — [Source](https://ar5iv.labs.arxiv.org/html/1910.08925)

- **STUN (Lee, Jung, Jo, 2022)**: Uses tabular Q-learning (not deep RL). Tunes 5 Linux kernel scheduler policies and 10 scheduler parameters automatically. Does NOT make per-process scheduling decisions; instead tunes global scheduler parameters. Showed 18.3% execution time improvement and 22.4% FPS improvement on face detection workloads. Published in Applied Sciences, MDPI. — [Source](https://www.mdpi.com/2076-3417/12/14/7072)

- **Reinforcement Learning for Scheduling Threads on a Multi-Core Processor (Shrivastava, 2010)**: Stanford CS229 project using tabular RL (not deep RL) to act as a meta-scheduler pinning threads to specific cores via OS API coremask. Very early work, not peer-reviewed. — [Source](https://cs229.stanford.edu/proj2010/Shrivastava-ReinforcementLearningForSchedulingThreadsOnAMultiCoreProcessor.pdf)

- **RL-OS-Scheduling (GitHub project, geekpradd)**: Uses RL to control quantum in MLFQ for process scheduling. Uses Q-learning to adapt time quantum and priority aging rate as a joint MDP. Formulates scheduling parameter regulation (not per-process scheduling) as MDP. — [Source](https://github.com/geekpradd/RL-OS-Scheduling)

- **PPO-for-CPU-Scheduling (Kyle-OD, GitHub/UD)**: University of Delaware OS course project using PPO for CPU scheduling simulation. Not a published paper. — [Source](https://github.com/Kyle-OD/PPO-for-CPU-Scheduling)

- **TuneAgent (arXiv:2508.12551)**: LLM agent with rule-based RL for Linux kernel tuning. Beats baselines by up to 5.6% overall performance. NOT per-process scheduling. — [Source](https://arxiv.org/pdf/2508.12551)

- **Scheduling of Time-Varying Workloads Using Reinforcement Learning (AAAI 2021)**: Uses RL for workload scheduling; state includes process meta-information (CPU/memory limits, scheduling priority). Reward based on negative slowdown of scheduled and waiting jobs. Published at AAAI 2021. — [Source](https://ojs.aaai.org/index.php/AAAI/article/view/17088/16895)

- **On the Impact of MDP Design for RL Agents in Resource Management (BRACIS 2021)**: Analyzes different MDP design choices for resource management. Finds that MDP design (state representation, reward shaping, action space) significantly impacts convergence and policy quality. Does not propose a new scheduler. — [Source](https://arxiv.org/pdf/2109.03202)

### Inferences
- The dominant approach for OS-level scheduling is parameter tuning (STUN, TuneAgent) rather than per-process decision-making RL.
- PPO is the most common deep RL algorithm applied to scheduling problems as of 2019–2026.
- No paper has combined MaskablePPO + GraphSAGE for per-CPU-process scheduling decisions with priority aging rewards in a simulator.

### Gaps
- No comprehensive survey paper specifically on deep RL for OS process-level (per-process, per-timeslice) scheduling was found.
- AAAI 2021 paper on time-varying workloads PDF was not parseable to extract full methodology details.

---

## What state representations are used?

### Takeaway
The dominant approach is flat feature vectors per job/process; graph-based representations are primarily used in cluster scheduling (Decima) and job-shop scheduling, not in OS process scheduling.

### Cited Findings

- RLScheduler uses per-job feature vectors: ID, user ID, group ID, executable ID, submission time, requested processors, runtime estimate, memory request. These are padded to 128 entries. — [Source](https://ar5iv.labs.arxiv.org/html/1910.08925)

- STUN observes global system metrics (workload type, utilization) not per-process state. It outputs global scheduler parameters not per-process actions. — [Source](https://www.mdpi.com/2076-3417/12/14/7072)

- DeepRM (Mao et al., 2016, HotNets): Visualizes state of resource occupancy and status of waiting jobs as an image; deep learning processes images, RL agent makes decisions. Uses REINFORCE policy gradient. Published at ACM HotNets 2016. — [Source](https://www.microsoft.com/en-us/research/publication/resource-management-deep-reinforcement-learning/)

### Inferences
- Image-based state representation (DeepRM) is an early approach that does not scale well to large job queues.
- Flat feature vectors per job (RLScheduler) are the most common approach but cannot capture structural relationships between jobs.
- Graph-based state representations that capture process relationships (as used in the target project via GraphSAGE) are largely absent from OS process scheduling literature.

### Gaps
- No paper found that uses observable history of process behavior (e.g., past CPU usage, wait time history) as part of the state representation in OS process scheduling.

---

## Have any systems integrated RL directly into the Linux kernel?

### Takeaway
STUN modifies Linux kernel scheduler parameters via Q-learning, but does not make per-process scheduling decisions. No system integrates deep RL directly into the Linux kernel scheduler's per-process decision logic.

### Cited Findings

- STUN modifies 5 Linux kernel scheduling policies and 10 parameters, using Q-learning. Runs as an external optimizer, not inside the kernel. — [Source](https://www.mdpi.com/2076-3417/12/14/7072)

- TuneAgent tunes Linux kernel parameters using LLM with RL, beating baselines by up to 5.6%. Not integrated into kernel. — [Source](https://arxiv.org/pdf/2508.12551)

- A 2025 paper "Rethinking Provenance Completeness with a Learning-Based Linux Scheduler" (arXiv:2510.08479) uses RL to predict task behaviors and make intelligent scheduling decisions. Full details not obtained due to PDF format. — [Source](https://arxiv.org/pdf/2510.08479)

### Inferences
- All practical RL + Linux kernel work operates as an external parameter-tuning layer, not as a replacement for the CFS scheduler's per-process decision logic.
- Simulation-based RL schedulers (including the target project) are the more common approach for learning per-process scheduling policies.

### Gaps
- The "Rethinking Provenance Completeness with a Learning-Based Linux Scheduler" paper (2025) could not be fully examined.

---

## What are the key limitations of RL-based OS scheduling approaches?

### Takeaway
Key limitations include: assumption of homogeneous resources, difficulty handling variable job/process counts, lack of graph structure in state representation, no action masking, no explicit fairness/priority, and training-test distribution mismatch.

### Cited Findings

- RLScheduler limitations: assumes homogeneous compute, caps observations at 128 jobs, no graph neural networks, no action masking, no fairness/priority mechanisms, no aging, trains offline only. — [Source](https://ar5iv.labs.arxiv.org/html/1910.08925)

- STUN limitations: static workload optimization only, Q-learning (not scalable to per-process decisions), does not handle dynamic process arrivals in real-time. — [Source](https://www.mdpi.com/2076-3417/12/14/7072)

- DeepRM limitations: image-based state does not scale to realistic cluster workloads with continuous job arrivals (acknowledged by RLScheduler paper). — [Source](https://ar5iv.labs.arxiv.org/html/1910.08925)

### Inferences
- The target project's combination of GraphSAGE + MaskablePPO + priority aging reward + variable process counts (1-8) + 500-seed statistical evaluation fills a gap not addressed by any existing OS/CPU scheduling RL paper.

### Gaps
- No paper combining all five features (GNN state, action masking, priority-aware reward, variable process count, statistical seeded evaluation) was found in the CPU/OS process scheduling domain.
