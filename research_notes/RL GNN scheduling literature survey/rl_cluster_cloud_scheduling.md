# RL-Based Cluster and Cloud Job Scheduling Systems

## Key papers on RL for cluster/cloud scheduling

### Takeaway
The landmark papers are DeepRM (2016), Decima (2019), RLScheduler (2019), Optimus (2018), and the MARL cluster scheduler (2021). These focus on cluster-level resource allocation, not CPU process-level scheduling, and use various RL algorithms + graph/image-based state representations.

### Cited Findings

**1. DeepRM: Resource Management with Deep Reinforcement Learning (Mao et al., ACM HotNets 2016)**
- Authors: H. Mao, M. Alizadeh, I. Menache, S. Kandula
- Venue: Proceedings of the 15th ACM Workshop on Hot Topics in Networks (HotNets), 2016
- RL Algorithm: REINFORCE (policy gradient)
- State: Image visualization of resource occupancy (CPU + memory as a 2D grid) and waiting job queue, processed as an image by CNN
- Action: Select which job to run next (discrete)
- Key claims: Deep RL can manage multi-resource multi-machine job scheduling better than SJF and Tetris heuristic under wide range of loads
- Limitations: Image-based state does not scale to realistic cluster workloads with continuous arrivals (acknowledged by successors). Simulation only, not validated on real cluster.
- Does NOT do: graph representations, action masking, CPU process scheduling, priority/fairness/aging, variable process counts
- Source: [Microsoft Research](https://www.microsoft.com/en-us/research/publication/resource-management-deep-reinforcement-learning/)

**2. Decima: Learning Scheduling Algorithms for Data Processing Clusters (Mao et al., ACM SIGCOMM 2019)**
- Authors: Hongzi Mao, Malte Schwarzkopf, Shaileshh Bojja Venkatakrishnan, Zili Meng, Mohammad Alizadeh (MIT CSAIL)
- Venue: ACM SIGCOMM 2019
- RL Algorithm: Policy gradient (REINFORCE variant)
- State: Job DAGs represented as graph; GNN produces node, job, and global embeddings. Custom GNN with two non-linear transformations (not standard GCN/GraphSAGE). Policy network: 6 FC layers [16,8,8,16,8,8] neurons for embeddings; 4 FC layers [32,16,8,1] neurons for action mapping.
- Action: (1) which node/stage to schedule, (2) how many executors to assign
- Action masking: Used to prevent re-selecting already-scheduled nodes
- Key claims: 21% improvement in avg JCT over hand-tuned heuristics on 25-node Spark cluster; 45% improvement over naive GNN-free approach; 30% total JCT improvement over SJF during peak load.
- Limitations: "Learns workload-specific policies whose effectiveness depends on whether broad test workload characteristics match training workload." Poor performance on edge cases. Cannot natively support advance reservations. Validated only on TPC-H query mixes on Spark. Training is slow due to REINFORCE variance.
- Does NOT do: CPU process scheduling, priority/fairness/aging, variable process counts (1-8), statistical evaluation across 500 seeds, GraphSAGE, observable history only
- Sources: [Decima PDF](https://zilimeng.com/papers/decima-sigcomm19.pdf); [MIT News](https://news.mit.edu/2019/decima-data-processing-0821); [Interpretable Decima](https://arxiv.org/html/2405.19131v1)

**3. Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters (Peng et al., EuroSys 2018)**
- Authors: Yanghua Peng, Yixin Bao, Yangrui Chen, Chuan Wu, Chuanxiong Guo
- Venue: EuroSys 2018
- Approach: NOT RL-based. Uses online resource-performance modeling (regression, not RL) to dynamically schedule resources to minimize JCT. Builds performance model per DL job based on observed throughput, then schedules resources optimally.
- Key claims: ~1.6x reduction in makespan vs dominant schedulers.
- Limitations: Requires training performance models per job type; not generalizable via RL.
- Does NOT do: RL, GNN, action masking, CPU process scheduling, priority/fairness, aging
- Sources: [EuroSys DOI](https://dl.acm.org/doi/10.1145/3190508.3190517); [Blog summary](https://blog.acolyer.org/2018/06/12/optimus-an-efficient-dynamic-resource-scheduler-for-deep-learning-clusters/)

**4. Large-scale ML Cluster Scheduling via Multi-agent Graph RL (Zhao & Wu, arXiv 2021)**
- Authors: Xiaoyang Zhao, Chuan Wu (University of Hong Kong)
- Venue: arXiv:2112.13354 (2021) — conference venue not confirmed
- RL Algorithm: Actor-critic policy gradient, MARL (each scheduler is an agent)
- State: Hierarchical GNN. Inner graph: nodes = CPUs, GPU groups, network switches with resource availability and placement status. Inter-scheduler graph: nodes = schedulers. Edge Condition Convolution (ECC) captures bandwidth and load.
- Action: Place worker/parameter server on local GPU group or forward to another scheduler
- Key claims: 20%+ improvement in avg JCT vs Tetris, load-balancing, LIF. Hierarchical GNNs generalize across Fat-tree, VL2, BCube topologies.
- Limitations: No resource preemption. Limited to 48 GPUs testbed. No action masking. No fairness/priority. Assumes synchronous training.
- Does NOT do: CPU process scheduling, action masking, priority/fairness, aging, variable 1-8 process counts, statistical evaluation across 500 seeds
- Source: [arXiv HTML](https://arxiv.org/html/2112.13354)

**5. RLScheduler: Automated HPC Batch Job Scheduler Using RL (Zhang et al., 2019)**
- Authors: Di Zhang, Dong Dai, Youbiao He, Forrest Sheng Bao, Bing Xie
- Venue: arXiv:1910.08925 (2019); later published at SC20
- RL Algorithm: PPO with actor-critic
- State: Per-job feature vectors (submission time, requested processors, runtime estimate, memory) padded to 128 entries. Kernel-based DNN applies 3-layer MLP to each job individually (permutation-equivariant by design).
- Action: Select which job from waiting queue to schedule next
- Key claims: Comparable to or better than state-of-the-art heuristics (SJF, WFP3, UNICEP, F1). Generalizes to unseen workloads with controlled degradation. Kernel DNN converges faster.
- Limitations: Homogeneous compute assumption. Caps at 128 jobs. No GNN. No action masking. No fairness/priority. No aging. Trains offline.
- Does NOT do: GNN/graph representation, action masking, CPU process-level scheduling, priority/fairness, aging, variable process counts (1-8), 500-seed statistical evaluation
- Source: [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/1910.08925)

**6. GNN-RL Multi-objective HPC Scheduling (Adimora et al., SC24)**
- Authors: Kyrian C. Adimora et al.
- Venue: SC24 (Supercomputing 2024) poster
- RL Algorithm: PPO
- GNN: Graph Attention Networks (GAT) — inductive, handles variable-size graphs
- State: Graph of jobs and resources with attention-weighted node features
- Key claims: 13.8–20.2% utilization improvement; 15.0–50.0% throughput improvement vs EASY-BF baseline on 377,234 real ALCF jobs. 76% effectiveness at 50K concurrent jobs.
- Limitations: Poster format; limited methodological detail available. Not specifically validated for CPU process-level scheduling.
- Does NOT do: CPU process scheduling, action masking, priority/fairness aging, variable 1-8 process counts, 500-seed statistical evaluation, GraphSAGE
- Source: [SC24 poster](https://sc24.supercomputing.org/proceedings/poster/poster_files/post137s2-file3.pdf)

### Inferences
- Decima is the most influential paper combining GNN + RL for scheduling, but operates on Spark DAGs (cluster-level), not CPU processes.
- No cluster scheduling paper uses GraphSAGE specifically; custom GNNs, GCN, GAT, and GIN are used.
- None of these papers address the OS process-level scheduling problem (arrival, burst time, priority, wait time, quantum-based scheduling).

### Gaps
- Scotch scheduler (as a named system) was not found in peer-reviewed literature; may refer to an internal system or unpublished work.

---

## Do any use action masking?

### Takeaway
Decima uses a limited form of action masking (preventing re-selection of scheduled nodes). The comprehensive action masking described in MaskablePPO (all invalid actions masked via logit manipulation) is not used in cluster scheduling papers.

### Cited Findings
- Decima: Uses action masking to prevent re-selecting scheduled nodes — not full MaskablePPO-style constraint enforcement. — [Source](https://arxiv.org/pdf/1810.01963)
- No other cluster/cloud scheduling paper in the literature uses action masking.

### Inferences
- The target project's use of MaskablePPO for systematic action masking (to prevent scheduling impossible processes given current state) is more comprehensive than Decima's limited masking.

---

## What reward functions are used?

### Takeaway
Most cluster/cloud schedulers optimize average job completion time (JCT) or makespan. Fairness is rarely incorporated into the reward function; priority aging is not used in any found paper.

### Cited Findings
- Decima: Minimizes average JCT. Reward = negative job completion time for each step. — [Source](https://zilimeng.com/papers/decima-sigcomm19.pdf)
- DeepRM: Minimizes average slowdown. Reward = negative count of active + waiting jobs per timestep. — [Source](https://www.microsoft.com/en-us/research/publication/resource-management-deep-reinforcement-learning/)
- RLScheduler: Reward = negative bounded slowdown of scheduled + waiting jobs. Can be adapted to other metrics. — [Source](https://ar5iv.labs.arxiv.org/html/1910.08925)
- MARL cluster (2021): Minimizes JCT with interference model. — [Source](https://arxiv.org/html/2112.13354)

### Inferences
- No cluster/cloud scheduling paper uses a soft priority + aging reward (reward shaped by process priority AND time waited, with priority weight increasing over time). This is a novel aspect of the target project.
