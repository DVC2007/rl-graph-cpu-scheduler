# Phase 1 findings: inventory and review

**Paper:** *Graph-Structured Reinforcement Learning for Fair and Efficient CPU Process Scheduling*
**Authors:** V. Saadhanaa, Disha Vishal Chaudhari, Sankari M. (SCOPE, VIT Chennai)
**Turnitin submission:** trn:oid:::3618:156082851. Both reports are for the same submission (6 pages, 3,183 words, 18,113 characters, main.pdf, 574.6 KB).

## 1. Files available

| File | What it is | Readable |
|---|---|---|
| `main_2_1.pdf` | The paper (6 pp, letter, built with XeLaTeX/xdvipdfmx and hyperref) | Yes: text, layout, and embedded images |
| `main_1_1.pdf` | Turnitin Similarity report (9 pp) | Yes |
| `Sujay_pravin_..._AI Writing report...pdf` | Turnitin AI-writing report (8 pp) | Yes |
| `attachment.txt` | The earlier humanizer workflow prompt | Yes (not part of the paper) |

**Not available:** the original `.tex` source, the `.bib`, the original figure files (vector or full-resolution), and any appendix. The paper itself has no appendix.

**Figure assets recovered from the PDF** (`pdfimages`, lossless PNG at embedded resolution):

| Fig | Content | Pixels | Embedded ppi |
|---|---|---|---|
| 1 | Architecture block diagram | — | Vector (drawn in LaTeX), so it must be recreated in TikZ |
| 2 (top) | Average waiting time bar charts | 1418×660 | 199 |
| 2 (bottom) | Starvation-rate bar charts | 1418×699 | 199 |
| 3 | Dashboard scheduling-state table and Gantt chart | 1459×342 | 418 |
| 4 | Dashboard dynamic process graph | 1459×502 | 418 |
| 5 | Contribution-decomposition chart | 1422×695 | 408 |

Figs. 2–5 are raster images and are printable at the sizes used. Higher quality would need the authors' original plot files.

## 2. Paper inventory

- **Sections:** Abstract, Index Terms; I Introduction (three contributions); II Related Work; III System Design (A Environment, B Reward Function, C Dynamic Process Graph, D GraphSAGE Feature Extractor); IV Experimental Setup (Uniform, Realistic, Baselines, Training); V Results (A Full Benchmark, B Contribution Decomposition, C Multi-Seed Robustness, D Visualisation); VI Discussion (A Limitations); VII Conclusion (including future work); References.
- **Equations:** an unnumbered observation vector **o**_i; Eq. (1) the reward; Eqs. (2)–(3) the GraphSAGE layers.
- **Tables:** I (full benchmark, 9 schedulers × 2 workloads × {Avg Wait, Starv.}); II (paired contribution decomposition); III (multi-seed evaluation).
- **Figures:** 1–5 as above.
- **References:** 20, numbered [1]–[20]. All are cited in the text except that [2] is cited only in the abstract.

## 3. Similarity report (4%): kept separate from the AI report

Seven match groups, all "Not Cited or Quoted"; 0 missing quotations or citations. Bibliography and quoted text are excluded.

| # | Source | Share | What matched |
|---|---|---|---|
| 1 | VIT Chennai student paper, 2026-07-24 | 2% | Author and affiliation block ("School of Computer Science and Engineering / Vellore Institute of Technology / Chennai, India", email domains) |
| 2 | dr.ntu.edu.sg | <1% | "To the best of our knowledge, … is the first … scheduler that" (Discussion) |
| 3 | nanocad.ee.ucla.edu | <1% | "To the best of our knowledge, no prior work has … a" (Introduction) |
| 4 | Rajagiri SET student paper, 2026-09-29 | <1% | "with two hidden layers of … units each" (Sec. III-D) |
| 5 | lnu.diva-portal.org | <1% | "the agent receives a … reward … the total … of" (Sec. III-B) |
| 6 | netlab.tkk.fi | <1% | "The Shortest Remaining Time … policy is … optimal for" (Introduction) |
| 7 | pds.ewi.tudelft.nl | <1% | "such as First-Come First-Served (FCFS) … the" (Abstract) |

**Assessment:** every match is boilerplate, an affiliation line, or a standard technical phrase. No substantive passage matches another work, and nothing needs a citation or quotation. Source 1 is worth one check: a VIT Chennai submission from July 2026 matches the affiliation block. If that was an earlier version of this paper or another paper from the same group, the overlap is expected.

## 4. AI-writing report (32%): kept separate from similarity

Turnitin reports 19 segments as "AI-generated" (32%), with no separate "AI-paraphrased" category. Its own caution says the score can be wrong and must not be the sole basis for any decision. These are the highlighted passages:

**Page 1 of the paper**
1. Abstract, from its opening through "…burst-history similarity among processes."
2. Abstract, "On exponential-burst (realistic) workloads…" to the end of the abstract.
3. Index Terms line.
4. Introduction paragraph 1, "CPU process scheduling remains…" through "…no explicit guarantee against starvation."
5. "Decima also lacks action masking and priority mechanisms."

**Page 2**
6. "Adimora et al. [11] applied Graph Attention Networks … at a finer granularity."
7. "Lassoued et al. [13] applied MaskablePPO … only 100 evaluation runs."
8. "Regarding fairness, we are not aware of any RL scheduler that learns aging behaviour from reward shaping."
9. Sec. III-B, "When any process exceeds a starvation threshold … grows linearly with the excess:" and Eq. (1).
10. "This formulation ensures that neglecting … without requiring explicit priority rules."
11. Sec. III-C opening, "At every tick, the graph is constructed from scratch … undirected edges:"
12. Edge-family items 1 (CPU Contention) and 2 (Type Similarity).

**Page 3**
13. "The starvation rate decreases from 87.3 % to 62.9 %, a reduction of 24.4 percentage points."
14. The whole "On realistic workloads, the average waiting time gap narrows…" paragraph.
15. "The same 500 workloads are evaluated … via bootstrap resampling."
16. "However, incorporating the graph reverses" (sentence fragment).
17. Part of the Table II caption ("denotes paired mean").
18. Multi-seed opening: "To assess reproducibility, five independently trained models are evaluated across 500 workloads each."

**Page 5**
19. Discussion: "an approximate proxy … hidden future information"; the 10.91-tick / 4.05-tick paragraph; the "starvation reduction of 12–24 percentage points … reward shaping" paragraph; Limitations through "/proc traces."; the first Conclusion sentence.

**Not flagged:** the DeepRM/RLScheduler and Decima paragraphs, the Huang & Ontañón sentence, Sec. III-A, III-D, IV, most of V-A/B, V-D, the figure captions, the second half of the Conclusion, and future work.

This list is only so the authors know where the report is looking. It is not evidence about how any passage was written.

## 5. Inconsistencies needing the authors' review (nothing changed)

**A. Numbers (most important)**
1. **Table I disagrees with Table II, Fig. 5, and the text on realistic workloads.**
   - Table I gives CFS at 28.33 and plain MaskablePPO at 28.28, so the flat agent is 0.05 *better* than CFS. Table II, Fig. 5, Sec. V-B, the abstract, and the Conclusion say the flat agent is +0.6 (+0.59) *worse*.
   - Table I gives the graph gain as 28.28 → 28.00 = −0.28. Table II and Fig. 5 give −0.7 (−0.74).
   - Table I gives the gap to SRTF as 28.00 − 17.32 = 10.68. Table II says −10.9, and the Discussion says "10.91".
   - Uniform-workload numbers all reconcile: −4.61, −0.76, −4.05; 49% / 8% / 43%.
   - **Likely cause:** Table II and Fig. 5 come from a different run, seed, or episode set than Table I. The authors need to say which numbers are canonical. This affects the paper's central claim that the flat RL agent "fails to significantly outperform CFS" on realistic workloads.
2. **Table III numbers (21.62 / 20.96) match neither workload in Table I** (uniform 19.30/18.54, realistic 28.28/28.00). The caption should say which workload or mix was used. Also, 21.62 − 20.96 = −0.66 against the reported −0.67. That is probably fine as a mean of paired differences, but worth confirming.
3. **What "Classical" means in Table II** isn't stated. It is FCFS for uniform workloads and CFS for realistic ones, per the Fig. 5 labels. One clause in the Table II caption or the text would fix this.

**B. Wording and claims**
4. **SRTF bound:** Sec. IV calls it a "theoretical upper bound", Fig. 2 calls it a "theoretical lower bound", and Table I says "theoretical bound". It is a lower bound on waiting time, which is an upper bound on achievable performance. Suggested wording for both places: "a theoretical lower bound on waiting time (oracle)".
5. **Fig. 3 caption vs image:** the caption says P4 (BACKGROUND) "receives priority upon exceeding the aging threshold", but the dashboard shows P4's WAIT = 0, so P4 never exceeded W* = 10. P3 (BACKGROUND, WAIT 32) did. Check which process was meant.
6. **Fig. 4 caption vs image:** the caption says three edge families are shown, but the image shows only type (dashed) and history (dotted) edges. "ready: []" means no CPU-contention edges exist in that snapshot. Either change the caption or use a snapshot that has ready processes.
7. **Embedded dashboard labels:** Fig. 3's image is titled "FIGURE 1. SCHEDULING STATE" and Fig. 4's image is titled "FIGURE 2. DYNAMIC PROCESS GRAPH". These conflict with the paper's numbering. Crop the titles out or regenerate the images.
8. **W1 dimension:** W1 ∈ ℝ^{32×20} implies a 20-dimensional input to layer 1. With 8 features per node, concatenating x_i with the neighbour mean gives 16, so W1 would be 32×16. W2 ∈ ℝ^{16×64} is consistent (2×32 → 16). Check whether the node features are 10-dimensional (for example, with a one-hot type encoding) and say so.
9. **"Starvation rate" is never defined in the text.** Only the Fig. 2 axis says "% processes waiting >10 ticks". One sentence in Sec. IV would fix this.
10. Body text spells the name "Ontáñón"; the correct spelling is Ontañón.

**C. References**
11. **[2]:** "Costa-juárez et al." could not be matched to sb3-contrib. Verified alternative: A. Raffin et al., "Stable-Baselines3: Reliable reinforcement learning implementations," *JMLR* 22(268):1–8, 2021, citing MaskablePPO in sb3-contrib.
12. **[12]:** the author is printed as "S. O. nón". Verified: S. Huang and S. Ontañón, FLAIRS vol. 35, 2022, doi:10.32473/flairs.v35i.130584.
13. **[13]:** "Y. Lassoued et al., ESWA vol. 263, 2026" could not be confirmed. The closest match is S. Lassoued, S. Lier, A. Schwung, arXiv:2601.09293 (2026), on maskable PPO with coloured timed Petri nets. The first initial is S, and no journal record was found.
14. **[14]:** arXiv:2512.10980 is actually A. Mamirov, "Reducing Fragmentation and Starvation in GPU Clusters through Dynamic Multi-Objective Scheduling" (2025). It is not "Anonymous, Aging-based fairness…". The text's claim that it uses aging as a hand-coded heuristic should be checked against that paper.
15. The rest ([1], [3]–[11], [15]–[20]) look plausible as printed but were not individually re-verified.

**D. Layout notes for reconstruction**
- The original was compiled with XeLaTeX, which is why section titles print in normal case rather than IEEE small caps. The reconstruction targets pdfLaTeX with IEEEtran, the standard Overleaf setup.
