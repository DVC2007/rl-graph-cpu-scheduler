# GNN/GCN/GraphSAGE Applied to Job/Task/Process Scheduling

## What graph construction strategies are used for scheduling?

### Takeaway
The dominant approach constructs a disjunctive graph (job-shop scheduling) or DAG (workflow/cluster scheduling) where nodes are tasks/operations and edges are dependencies or resource contention relationships. Bipartite job-machine graphs are also used.

### Cited Findings

- **Learning to Dispatch for Job Shop Scheduling via Deep RL (L2D, Zhang et al., NeurIPS 2020)**: Uses disjunctive graph representation of JSSP. Nodes = operations, edges = conjunctive (precedence) and disjunctive (machine contention). GNN architecture: Graph Isomorphism Network (GIN) combined with PPO. Key claim: size-agnostic policy enables generalization to large-scale instances. — [Source](https://nips.cc/virtual/2020/public/poster_11958dfee29b6709f48a9ba0387a2431.html)

- **Geometric Deep RL for Dynamic DAG Scheduling (Grinsztajn et al., arXiv 2020)**: Uses DAG of tasks. Nodes = task instances with features (successor count, predecessor count, task type, availability, running status, critical path). Edges = data dependencies. GNN: GCN (graph convolutional networks) with stacked layers. RL: A2C (synchronous actor-critic). Applied to Cholesky factorization in HPC. — [Source](https://arxiv.org/html/2011.04333v1)

- **Decima (Mao et al., SIGCOMM 2019)**: Uses job DAGs. Nodes = job stages, edges = stage dependencies. Custom GNN with non-linear aggregation (not standard GCN/GraphSAGE). Claim: "adding a second non-linear transformation g(·) is critical for learning strong scheduling policies." — [Source](https://arxiv.org/pdf/1810.01963)

- **GNN-RL for HPC scheduling (Adimora et al., SC24)**: Uses Graph Attention Networks (GAT). Cited improvement: 13.8–20.2% utilization improvement and 15.0–50.0% throughput improvement vs EASY-BF baseline on 377,234 real ALCF jobs. GAT is inductive and generalizes to graphs of variable size. — [Source](https://sc24.supercomputing.org/proceedings/poster/poster_files/post137s2-file3.pdf)

- **Graph Neural Networks for Job Shop Scheduling Problems: A Survey (arXiv 2406.14096, 2024)**: Comprehensive survey of GNN approaches for JSSP. Notes disjunctive graph, DAG, and operation-machine bipartite graph as the three dominant representations. — [Source](https://arxiv.org/pdf/2406.14096)

- **Multi-agent Graph RL for ML Cluster Scheduling (Zhao & Wu, arXiv 2021)**: Uses hierarchical GNNs. Inner graph: nodes = CPUs, GPU groups, switches. Inter-scheduler graph: nodes = schedulers. Edges encode network bandwidth. Uses Edge Condition Convolution (ECC). Actor-critic RL. — [Source](https://arxiv.org/html/2112.13354)

### Inferences
- The "process graph" in the target project (where nodes = processes with features like burst time, wait time, priority; edges = relationships between co-running processes) has no direct analog in the literature. Most GNN schedulers use task-dependency (DAG) graphs, not process-state graphs.
- GraphSAGE specifically is not used in the core scheduling literature—GAT, GCN, and GIN are more common, along with custom GNN variants.

### Gaps
- No paper found that constructs a graph over CPU processes (rather than jobs with dependencies) and applies GraphSAGE to embed individual process states.

---

## What GNN architectures are used and why?

### Takeaway
GCN, GIN, GAT, and custom message-passing GNNs are dominant; GraphSAGE is notably absent from the main scheduling literature but provides a key advantage: inductive learning with neighborhood sampling, enabling generalization to unseen graph sizes.

### Cited Findings

- L2D (NeurIPS 2020) uses GIN (Graph Isomorphism Network) combined with PPO. GIN is theoretically more expressive than GCN for graph structure discrimination. — [Source](https://nips.cc/virtual/2020/public/poster_11958dfee29b6709f48a9ba0387a2431.html)

- Geometric DAG scheduling (arXiv 2020) uses GCN with stacked layers. w layers used = window size parameter. — [Source](https://arxiv.org/html/2011.04333v1)

- Decima uses custom GNN with two non-linear transformations at each node. Not standard GCN or GraphSAGE. — [Source](https://arxiv.org/pdf/1810.01963)

- GNN-RL HPC (SC24) uses GAT (Graph Attention Networks). GAT is "inductive and generalizes to graphs of variable size and topology, which is essential because the job queue changes at every scheduling step." — [Source](https://sc24.supercomputing.org/proceedings/poster/poster_files/post137s2-file3.pdf)

- GraphSAGE (Hamilton, Ying, Leskovec, NeurIPS 2017): Generates embeddings by sampling and aggregating features from a node's local neighborhood. Key property: inductive framework—generates embeddings for previously unseen nodes. Published NeurIPS 2017. — [Source](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)

- **Learning to schedule job-shop problems (arXiv 2106.01086)**: Notes that "Because GNN enables the scheduling policy to handle differently-sized graphs, the trained policy can be used for problem instances in different sizes." Does not specify which GNN architecture (GCN, GraphSAGE, etc.). — [Source](https://arxiv.org/pdf/2106.01086)

### Inferences
- GraphSAGE's inductive nature (neighborhood sampling and aggregation) makes it theoretically well-suited for CPU process scheduling where the number of processes varies (1-8 in the target project), but this specific combination has not appeared in the literature.
- GAT and GIN appear in more recent work; GCN in older work. The choice of GraphSAGE for the target project is novel in the CPU scheduling context.

### Gaps
- No paper found that uses GraphSAGE specifically for any scheduling problem (CPU, cluster, or job-shop).

---

## Do GNNs handle variable graph sizes? How?

### Takeaway
Most GNN scheduling systems handle variable graph sizes through the inherent permutation-invariance and inductive properties of GNNs, using global pooling or per-node action selection. The target project's 1-8 variable process count is handled similarly but at CPU process granularity.

### Cited Findings

- L2D (NeurIPS 2020): Uses "size-agnostic policy network that enables generalization on large-scale instances." The GNN processes graphs of any size; policy selects from per-node embeddings. — [Source](https://nips.cc/virtual/2020/public/poster_11958dfee29b6709f48a9ba0387a2431.html)

- Geometric DAG scheduling (arXiv 2020): Key claim: "strong transfer learning capabilities across different problem sizes and resource counts." Tested on different scales. — [Source](https://arxiv.org/html/2011.04333v1)

- Decima: Policy outputs actions (node selection + executor count) from per-node embeddings. Handles varying number of job stages. But requires fixed action set; action masking used to prevent re-selecting scheduled nodes. — [Source](https://arxiv.org/pdf/1810.01963)

- GNN-RL HPC (SC24): "maintains 76% effectiveness at 50K concurrent jobs." Uses GAT which is size-agnostic by design. — [Source](https://sc24.supercomputing.org/proceedings/poster/poster_files/post137s2-file3.pdf)

### Inferences
- The target project's variable process count (1-8) is a simpler variant of the variable graph size problem solved in cluster/job-shop literature, but is novel in the CPU/OS process scheduling context.
- None of these papers specifically address CPU processes (with arrival times, burst times, priorities, wait times) as graph nodes.

### Gaps
- No paper found that combines inductive GNN (GraphSAGE) with action masking for variable-count CPU processes in a simulator.

---

## What are the key limitations of GNN-based schedulers?

### Takeaway
Key limitations: computational overhead of GNN inference, training instability with sparse rewards, assumption of known graph structure, limited generalization to fundamentally different workload distributions, and none address CPU OS process scheduling.

### Cited Findings

- Geometric DAG scheduling (arXiv 2020): "Higher computational overhead than heuristics during inference." "Evaluated only on Cholesky factorization." Does not address CPU process scheduling, action masking, priority/fairness. — [Source](https://arxiv.org/html/2011.04333v1)

- L2D (NeurIPS 2020): Problem-instance-specific training; does not generalize across fundamentally different problem classes. No action masking, no priority/fairness. — [Source](https://nips.cc/virtual/2020/public/poster_11958dfee29b6709f48a9ba0387a2431.html)

- Decima: "Learns workload-specific policies whose effectiveness depends on whether broad test workload characteristics match the training workload." "Does not natively support advance reservations." Poor performance on edge cases. — [Source](https://arxiv.org/html/2405.19131v1)

- Multi-agent cluster GNN (arXiv 2021): "No resource preemption—jobs run to completion without dynamic resource adjustment." "Limited to 48 GPUs." No action masking, no priority/fairness. — [Source](https://arxiv.org/html/2112.13354)

### Inferences
- The target project's use of 500 seeded workloads for statistical evaluation directly addresses the reproducibility/evaluation limitations of most GNN scheduling papers, which typically report results on limited trace datasets without rigorous statistical evaluation.
