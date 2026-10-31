# Action Masking in RL Schedulers and Priority/Fairness-Aware RL Schedulers

## What papers use action masking in RL for scheduling? How is it implemented?

### Takeaway
Action masking in RL schedulers is an emerging technique (2020–2026). The canonical theoretical treatment is Huang & Ontañón (FLAIRS 2022). MaskablePPO in sb3-contrib is the standard implementation library. A 2026 job-shop paper uses MPPO with Petri net-based masking. No paper uses MaskablePPO for CPU process scheduling.

### Cited Findings

**1. A Closer Look at Invalid Action Masking in Policy Gradient Algorithms (Huang & Ontañón)**
- Authors: Shengyi Huang, Santiago Ontañón
- Venue: International FLAIRS Conference Proceedings, Vol. 35 (2022); arXiv:2006.14171 (2020)
- Year: 2020/2022
- RL Algorithm: PPO (tested environment: μRTS strategy game)
- Masking mechanism: Replaces logits of invalid actions with large negative constant (e.g., M = -1×10⁸), then applies softmax. This mathematically ensures invalid actions receive near-zero probability and valid actions are renormalized.
- Theoretical justification: "Proposition 1" establishes gradients produced by invalid action masking constitute valid policy gradients for the masked policy π'_θ. The masking function is state-dependent and differentiable.
- Key comparison: Masking vs. penalty-based invalid action handling. On 10×10 μRTS map, penalty-based approach required 3.43% of training time to achieve first reward vs. 0.06% for masking. Penalty hyperparameter tuning difficult; excessive penalties suppress exploration.
- Limitations: Only tested on μRTS (single environment). Only PPO tested. Partial masking used (not all invalid action dimensions masked).
- Does NOT do: scheduling application, CPU process scheduling, priority/fairness
- Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/2006.14171)

**2. Implementing Action Mask in PPO Algorithm (Xiao et al., ICT Express 2020)**
- Venue: ICT Express (Elsevier), 2020
- RL Algorithm: PPO with action mask
- Masking mechanism: Boolean mask passed to environment; environment provides valid/invalid indicator per action. Both agent and environment blocks modified. Agent ignores invalid actions during sampling.
- Key claims: "Algorithm yields much higher return with a moderate number of training steps" vs standard PPO.
- Limitations: Not applied to scheduling specifically in this paper.
- Source: [ScienceDirect](https://www.sciencedirect.com/article/pii/S2405959520300746)

**3. Policy-Based RL with Action Masking for Dynamic Job Shop Scheduling (Lassoued et al., arXiv 2601.09293, 2026)**
- Authors: Sofiene Lassoued, Stefan Lier, Andreas Schwung
- Venue: Expert Systems with Applications (journal, 2026)
- RL Algorithm: Maskable PPO (MPPO), explicitly referencing sb3-contrib
- State representation: Colored-Timed Petri Nets (CTPN) — tokens encode job attributes; places represent system state
- What gets masked: Invalid job-machine transitions (precedence constraints, unavailable machines, incompatible operations)
- Two masking strategies: (1) Gradient-free masking: -∞ logits for invalid actions via Petri net guard functions; (2) Gradient-based masking (novel): penalty term λ_inv × Σ π_θ(a|s) for invalid actions — agent learns to suppress infeasible moves
- Key claims: "Combines CTPN with MPPO RL enabling adaptive scheduling under uncertainties"; handles machine failures (Weibull distribution) and stochastic job arrivals (Gamma distribution)
- Limitations: Sparse reward challenge. Training is problem-instance-specific (no generalization across sizes). No comparison with recent DRL baselines.
- Does NOT do: CPU process scheduling, priority/fairness/aging, variable process counts (1-8), 500-seed statistical evaluation, graph neural network state representation
- Source: [arXiv](https://arxiv.org/html/2601.09293v1)

**4. MaskablePPO in stable-baselines3-contrib (sb3-contrib)**
- Implementation: MaskablePPO in sb3-contrib library. action_masks parameter is a boolean array (True=valid, False=invalid). Modifies policy's action distribution to incorporate mask constraints. Automatically retrieves masks via ActionMasker wrapper or manual provision.
- Technical mechanism: "Sets probabilities corresponding to infeasible actions to negative infinity, then re-normalizes using Softmax, resulting in infeasible actions being assigned null probabilities while feasible action probabilities sum to one."
- Key difference from standard PPO: collect_rollouts() accepts use_masking boolean; get_distribution() accepts action_masks.
- Limitation: Recurrent policies unsupported. SubprocVecEnv requires environments implementing action_masks internally.
- Source: [SB3-Contrib Docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)

**5. Action Space Shaping in Deep Reinforcement Learning (Kanervisto et al., IEEE CoG 2020)**
- Authors: Anssi Kanervisto, Christian Scheller, Ville Hautamäki
- Venue: IEEE Conference on Games (CoG), 2020; arXiv:2004.00980
- Contribution: Broader framework of "action space shaping" including action removal (masking), action combination, and action discretization. Foundational work that contextualizes masking within a larger space of action space management strategies.
- Does NOT do: scheduling application, GNN, priority/fairness
- Source: [Semantic Scholar](https://www.semanticscholar.org/paper/A-Closer-Look-at-Invalid-Action-Masking-in-Policy-Huang-Ontan%C3%A9n/83b7d091fa44906d4cc873fdfaa4ba3d1858e7bb)

### Inferences
- The use of MaskablePPO (sb3-contrib) for CPU process scheduling is novel: the 2026 job-shop paper (arXiv 2601.09293) is the closest, but uses job-shop manufacturing domain, not OS process scheduling.
- No paper combines MaskablePPO + GraphSAGE for any scheduling problem.

### Gaps
- No paper found that applies MaskablePPO to CPU/OS process scheduling specifically.

---

## What papers address priority or fairness explicitly in the RL reward?

### Takeaway
Explicit priority or fairness in RL reward functions is rare in scheduling literature. Most schedulers optimize aggregate efficiency metrics (JCT, makespan). Papers on aging/starvation prevention in RL are extremely rare. Fairness-aware RL schedulers are mostly from wireless networking and GPU cluster domains, not CPU process scheduling.

### Cited Findings

**1. Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads (Narayanan et al., OSDI 2020)**
- Authors: Deepak Narayanan et al.
- Venue: OSDI 2020
- Approach: Not RL. Uses fairness metrics (per-job weighted fair sharing, dominant resource fairness) within a traditional optimizer, not RL. Shows fairness and efficiency tradeoffs in GPU cluster scheduling.
- Relevant finding: Fairness policy design is complex; RL not used here.
- Source: [USENIX](https://www.usenix.org/system/files/osdi20-narayanan_deepak.pdf)

**2. Towards Fair and Firm Real-Time Scheduling via RL (arXiv 2403.00766)**
- Venue: arXiv 2024
- Approach: RL with SLO-aware reward for DNN multi-tenant multi-accelerator systems. Explicitly models per-tenant fairness in reward. "87% of tenants meeting their firm real-time requirements compared to 60% with SLA-unaware RL baselines."
- Reward includes explicit fairness component for tenant SLO satisfaction.
- Does NOT do: CPU process scheduling, aging, GraphSAGE, action masking
- Source: [arXiv](https://arxiv.org/pdf/2403.00766)

**3. Reducing Fragmentation and Starvation in GPU Clusters via Dynamic Multi-objective Scheduling (arXiv 2512.10980)**
- Venue: arXiv 2025
- Approach: Hybrid Priority Scheduler (HPS) — NOT RL. Combines efficiency bias, aging-based fairness, and GPU-blocking mitigation into a scoring model. Aging: "gradually increasing the effective priority of processes that have been waiting." Key insight: aging is used as an anti-starvation mechanism in a multi-objective scoring model.
- Does NOT do: RL, GNN, action masking, CPU process scheduling
- Source: [arXiv](https://arxiv.org/pdf/2512.10980)

**4. Reinforcement learning based multi-objective task scheduling for energy efficient and cost effective cloud edge computing (Nature Scientific Reports 2025)**
- Venue: Nature Scientific Reports, 2025
- Approach: Multi-objective RL with reward = weighted combination of waiting time, energy efficiency, cost. Fairness handled via reward weighting.
- Limitations: No explicit priority aging. Cloud/edge computing domain, not CPU process scheduling.
- Source: [Nature](https://www.nature.com/articles/s41598-025-25666-1)

**5. Decima (SIGCOMM 2019)**: No explicit priority or fairness in reward. Solely minimizes avg JCT. Fairness is not considered. — [Source](https://zilimeng.com/papers/decima-sigcomm19.pdf)

**6. RLScheduler (2019)**: Reward = negative bounded slowdown. No explicit priority. Fairness explored only post-hoc as an analysis metric, not in the reward function. — [Source](https://ar5iv.labs.arxiv.org/html/1910.08925)

### Inferences
- The target project's "soft priority + aging reward" (reward shaped by both job priority AND time-since-arrival, with priority weight escalating over wait time) is not replicated in any found paper. This directly addresses starvation prevention in a way that no RL scheduler paper explicitly claims to do.
- The closest analog is the GPU cluster HPS approach (arXiv 2512.10980) which uses aging, but this is a non-RL heuristic scoring system, not RL.

### Gaps
- No paper found that combines (a) explicit job/process priority in RL reward with (b) aging mechanism that increases priority weight over wait time, applied to (c) CPU process scheduling.
- Priority aging in RL (as distinct from aging in heuristic schedulers) appears entirely absent from the peer-reviewed literature.

---

## What is the theoretical justification for action masking vs. penalty-based approaches?

### Takeaway
Huang & Ontañón (FLAIRS 2022) provide the key theoretical result: action masking produces valid policy gradients for the masked policy, while reward penalties do not guarantee gradient correctness and perform poorly at scale.

### Cited Findings
- Huang & Ontañón (2022): Masking zeroes gradient components for masked actions. Naive masking (sampling from masked distribution but computing gradients from unmasked distribution) causes KL divergence explosion. Penalty-based methods require difficult hyperparameter tuning and fail on large action spaces. — [Source](https://ar5iv.labs.arxiv.org/html/2006.14171)

- Reward penalty comparison: On 10×10 μRTS, penalty-based: 3.43% training time to first reward; masking: 0.06% training time to first reward. Scaling advantage grows with action space size. — [Source](https://ar5iv.labs.arxiv.org/html/2006.14171)

- sb3-contrib MaskablePPO: Technically implements masking by setting infeasible logits to -∞ before softmax, producing valid gradient signal only for feasible actions. — [Source](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)

### Inferences
- The target project's use of MaskablePPO is theoretically justified by Huang & Ontañón's gradient correctness proof. For CPU process scheduling with variable process counts (1-8), action masking prevents selection of "empty process slots" — an invalid action that penalty-based methods would handle poorly.

---

## Statistical evaluation methodology in RL scheduling papers

### Takeaway
Most RL scheduling papers use limited statistical evaluation — typically 10 random trials per configuration or trace-based evaluation without explicit seeding. The target project's 500 seeded workloads is significantly more rigorous than any found paper.

### Cited Findings
- RLScheduler: 10 random trials per configuration. — [Source](https://ar5iv.labs.arxiv.org/html/1910.08925)
- Policy-Based RL with Action Masking for DJSS (2026): 100 independent runs per configuration with 95% CI. — [Source](https://arxiv.org/html/2601.09293v1)
- Decima: Evaluated on TPC-H query mixes on single 25-node Spark cluster. No explicit seeded evaluation reported. — [Source](https://zilimeng.com/papers/decima-sigcomm19.pdf)
- GNN-RL HPC (SC24): 377,234 real ALCF jobs — real trace, not seeded synthetic workloads. — [Source](https://sc24.supercomputing.org/proceedings/poster/poster_files/post137s2-file3.pdf)

### Inferences
- 100 runs (Lassoued 2026) is the highest found for action-masking RL scheduling. 500 seeded workloads in the target project exceeds all found papers in systematic seeded evaluation rigor.
- Seeded synthetic evaluation (vs. single real trace) enables more controlled statistical comparison.
