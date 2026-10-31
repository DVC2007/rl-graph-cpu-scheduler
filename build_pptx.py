"""Build conference presentation for GraphSAGE + MaskablePPO CPU Scheduling paper."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

FIGURES = os.path.join(os.path.dirname(__file__), "paper_humanized", "figures")
OUT = os.path.join(os.path.expanduser("~"), "Downloads", "Conference_Presentation.pptx")

# Color palette
DARK_BLUE = RGBColor(0x1B, 0x3A, 0x5C)
ACCENT_BLUE = RGBColor(0x2E, 0x86, 0xC1)
ACCENT_GREEN = RGBColor(0x27, 0xAE, 0x60)
ACCENT_ORANGE = RGBColor(0xE6, 0x7E, 0x22)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xF2, 0xF3, 0xF4)
DARK_GRAY = RGBColor(0x2C, 0x3E, 0x50)
MED_GRAY = RGBColor(0x7F, 0x8C, 0x8D)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

W = prs.slide_width
H = prs.slide_height


def add_bg(slide, color=WHITE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
    else:
        shape.line.fill.background()
    return shape


def add_text_box(slide, left, top, width, height, text, font_size=18,
                 bold=False, color=DARK_GRAY, alignment=PP_ALIGN.LEFT,
                 font_name="Calibri"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_bullet_slide_content(tf, items, font_size=20, color=DARK_GRAY, spacing=Pt(8)):
    """Add multiple bullet points to an existing text frame."""
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = spacing
        p.level = 0


def add_sub_bullet(tf, text, font_size=16, color=MED_GRAY):
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.name = "Calibri"
    p.level = 1
    p.space_after = Pt(4)


def make_content_slide(title_text, has_accent_bar=True):
    """Create a slide with standard header bar + title."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    add_bg(slide, WHITE)
    # Top accent bar
    if has_accent_bar:
        add_rect(slide, Inches(0), Inches(0), W, Inches(0.06), ACCENT_BLUE)
    # Title
    add_text_box(slide, Inches(0.7), Inches(0.3), Inches(11.5), Inches(0.7),
                 title_text, font_size=32, bold=True, color=DARK_BLUE)
    # Thin separator line
    add_rect(slide, Inches(0.7), Inches(1.0), Inches(2.0), Inches(0.04), ACCENT_BLUE)
    return slide


# ============================================================
# SLIDE 1: TITLE
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
# Big blue band at top
add_rect(slide, Inches(0), Inches(0), W, Inches(4.2), DARK_BLUE)
# Accent stripe
add_rect(slide, Inches(0), Inches(4.2), W, Inches(0.08), ACCENT_BLUE)

add_text_box(slide, Inches(0.8), Inches(0.8), Inches(11.5), Inches(1.2),
             "Graph-Structured Reinforcement Learning for\nFair and Efficient CPU Process Scheduling",
             font_size=36, bold=True, color=WHITE)

add_text_box(slide, Inches(0.8), Inches(2.4), Inches(11.5), Inches(0.5),
             "GraphSAGE + MaskablePPO with Soft-Priority Aging Reward",
             font_size=22, color=RGBColor(0xAE, 0xD6, 0xF1))

add_text_box(slide, Inches(0.8), Inches(3.2), Inches(11.5), Inches(0.6),
             "V. Saadhanaa  |  Disha Vishal Chaudhari  |  Sankari M.",
             font_size=20, bold=True, color=WHITE)

add_text_box(slide, Inches(0.8), Inches(3.7), Inches(11.5), Inches(0.4),
             "School of Computer Science and Engineering, Vellore Institute of Technology, Chennai",
             font_size=16, color=RGBColor(0xD5, 0xDB, 0xDB))

# Bottom section — keywords
add_text_box(slide, Inches(0.8), Inches(5.0), Inches(11.5), Inches(0.8),
             "Process Scheduling  |  Starvation Prevention  |  Graph Neural Networks  |  Reinforcement Learning  |  Fairness",
             font_size=16, color=MED_GRAY, alignment=PP_ALIGN.LEFT)

# ============================================================
# SLIDE 2: PROBLEM & MOTIVATION
# ============================================================
slide = make_content_slide("Problem & Motivation")

txBox = slide.shapes.add_textbox(Inches(0.7), Inches(1.3), Inches(5.8), Inches(5.5))
tf = txBox.text_frame
tf.word_wrap = True
add_bullet_slide_content(tf, [
    "CPU scheduling is a decades-old problem, yet starvation persists",
    "SRTF is provably optimal for avg wait — but needs future knowledge",
    "CFS (Linux default) is fast & elegant, but has a blind spot:",
], font_size=20)
add_sub_bullet(tf, "Nothing says \"this process has been waiting too long, run it\"", font_size=17)
tf.add_paragraph()  # spacer
add_bullet_slide_content(tf, [
    "Prior RL schedulers (DeepRM, Decima, RLScheduler) work at cluster/job level — not OS process level",
    "No prior work combines GNN + action masking + fairness-aware reward for per-process scheduling",
], font_size=20)

# Right side — key stat callout box
box = add_rect(slide, Inches(7.2), Inches(1.5), Inches(5.2), Inches(2.0),
               RGBColor(0xEB, 0xF5, 0xFB), ACCENT_BLUE)
add_text_box(slide, Inches(7.5), Inches(1.7), Inches(4.7), Inches(0.5),
             "The Gap We Address", font_size=20, bold=True, color=ACCENT_BLUE)
add_text_box(slide, Inches(7.5), Inches(2.2), Inches(4.7), Inches(1.1),
             "CFS starvation rate: 87.3%\nOur model: 62.9%\n= 24.4 percentage point reduction",
             font_size=18, color=DARK_GRAY)

# Another callout
box2 = add_rect(slide, Inches(7.2), Inches(4.0), Inches(5.2), Inches(2.0),
                RGBColor(0xFD, 0xED, 0xEC), ACCENT_ORANGE)
add_text_box(slide, Inches(7.5), Inches(4.2), Inches(4.7), Inches(0.5),
             "Key Insight", font_size=20, bold=True, color=ACCENT_ORANGE)
add_text_box(slide, Inches(7.5), Inches(4.7), Inches(4.7), Inches(1.1),
             "Without the graph, a flat RL agent cannot\noutperform CFS on realistic workloads.\nThe relational structure is essential.",
             font_size=18, color=DARK_GRAY)

# ============================================================
# SLIDE 3: CONTRIBUTIONS
# ============================================================
slide = make_content_slide("Our Contributions")

# Three contribution boxes side by side
for i, (title, body, accent) in enumerate([
    ("Dynamic Process Graph",
     "Reconstructed every tick from\nobservable features.\n\n3 edge families:\n- CPU contention\n- Type similarity\n- Burst-history similarity",
     ACCENT_BLUE),
    ("GraphSAGE + MaskablePPO",
     "GraphSAGE: designed for graphs\nwhose topology changes between\nforward passes.\n\nAction masking: invalid actions\nset to -inf before softmax.\n57x faster convergence.",
     ACCENT_GREEN),
    ("Soft-Priority Aging Reward",
     "Penalty grows linearly when\nwaiting exceeds threshold W*.\n\nThe agent learns anti-starvation\nbehaviour purely from reward\nshaping — no hand-coded rules.",
     ACCENT_ORANGE),
]):
    left = Inches(0.7 + i * 4.1)
    # Colored top bar on card
    add_rect(slide, left, Inches(1.4), Inches(3.7), Inches(0.06), accent)
    # Card background
    add_rect(slide, left, Inches(1.46), Inches(3.7), Inches(5.0),
             LIGHT_GRAY, RGBColor(0xD5, 0xD8, 0xDC))
    # Card title
    add_text_box(slide, left + Inches(0.2), Inches(1.6), Inches(3.3), Inches(0.6),
                 title, font_size=22, bold=True, color=accent)
    # Card body
    add_text_box(slide, left + Inches(0.2), Inches(2.3), Inches(3.3), Inches(3.8),
                 body, font_size=17, color=DARK_GRAY)

# ============================================================
# SLIDE 4: SYSTEM ARCHITECTURE
# ============================================================
slide = make_content_slide("System Architecture")

# Left side — architecture description
txBox = slide.shapes.add_textbox(Inches(0.7), Inches(1.3), Inches(5.5), Inches(5.5))
tf = txBox.text_frame
tf.word_wrap = True
add_bullet_slide_content(tf, [
    "Single-CPU, up to 8 concurrent processes",
    "Each process: CPU bursts + I/O wait periods",
    "Observation: 64-dim vector (8 features x 8 slots)",
], font_size=20)
add_sub_bullet(tf, "wait time, type, CPU used, schedule count, burst stats, ready flag, blocked time")
tf.add_paragraph()
add_bullet_slide_content(tf, [
    "Action space: Discrete(N+1) — pick a process or idle",
    "GraphSAGE: 2 layers, mean-pooling aggregation",
], font_size=20)
add_sub_bullet(tf, "W1 in R^{32x20}, W2 in R^{16x64} -> 48-dim embedding")
tf.add_paragraph()
add_bullet_slide_content(tf, [
    "PPO actor-critic: 2 hidden layers of 64 units each",
    "Graph rebuilt from scratch every tick — no state carryover",
], font_size=20)

# Reward equation box
box = add_rect(slide, Inches(6.8), Inches(1.3), Inches(5.8), Inches(2.5),
               RGBColor(0xFE, 0xF9, 0xE7), ACCENT_ORANGE)
add_text_box(slide, Inches(7.1), Inches(1.5), Inches(5.2), Inches(0.5),
             "Reward Function", font_size=22, bold=True, color=ACCENT_ORANGE)
add_text_box(slide, Inches(7.1), Inches(2.1), Inches(5.2), Inches(1.5),
             "r_t = - SUM [ w_type + alpha * max(0, W_i - W*) ]\n\nalpha = 0.5, W* = 10 ticks\n\nLonger a process waits past threshold,\nthe costlier it becomes to ignore it.",
             font_size=16, color=DARK_GRAY)

# Architecture pipeline box
box2 = add_rect(slide, Inches(6.8), Inches(4.2), Inches(5.8), Inches(2.5),
                RGBColor(0xEB, 0xF5, 0xFB), ACCENT_BLUE)
add_text_box(slide, Inches(7.1), Inches(4.4), Inches(5.2), Inches(0.5),
             "Pipeline (every tick)", font_size=22, bold=True, color=ACCENT_BLUE)
add_text_box(slide, Inches(7.1), Inches(5.0), Inches(5.2), Inches(1.5),
             "Observation (R^64)\n    -> Graph Builder (V, E)\n        -> GraphSAGE (2 layers)\n            -> Projection (R^48)\n                -> Actor MLP + Action Mask -> pi(a|s)\n                -> Critic MLP -> V(s)",
             font_size=15, color=DARK_GRAY, font_name="Consolas")

# ============================================================
# SLIDE 5: DYNAMIC PROCESS GRAPH
# ============================================================
slide = make_content_slide("Dynamic Process Graph")

txBox = slide.shapes.add_textbox(Inches(0.7), Inches(1.3), Inches(5.5), Inches(5.5))
tf = txBox.text_frame
tf.word_wrap = True
add_bullet_slide_content(tf, [
    "Entire graph discarded & rebuilt every tick",
    "Processes = nodes; 3 edge families:",
], font_size=20)
add_sub_bullet(tf, "CPU Contention: both processes in ready queue")
add_sub_bullet(tf, "Type Similarity: same priority class")
add_sub_bullet(tf, "Burst-History: avg burst lengths differ by <= 1 tick")
tf.add_paragraph()
add_bullet_slide_content(tf, [
    "All edges include self-loops",
    "No state carries over between ticks",
    "GraphSAGE aggregates neighbourhood via mean-pooling",
], font_size=20)
tf.add_paragraph()
add_bullet_slide_content(tf, [
    "Key advantage: captures relational context a flat vector misses",
], font_size=20, color=ACCENT_BLUE)

# Process graph screenshot
slide.shapes.add_picture(
    os.path.join(FIGURES, "fig4_process_graph.png"),
    Inches(6.8), Inches(1.3), Inches(5.8))

# ============================================================
# SLIDE 6: EXPERIMENTAL SETUP
# ============================================================
slide = make_content_slide("Experimental Setup")

# Left column — workloads
add_text_box(slide, Inches(0.7), Inches(1.3), Inches(5.5), Inches(0.5),
             "Workloads", font_size=24, bold=True, color=ACCENT_BLUE)
txBox = slide.shapes.add_textbox(Inches(0.7), Inches(1.9), Inches(5.5), Inches(2.5))
tf = txBox.text_frame
tf.word_wrap = True
add_bullet_slide_content(tf, [
    "Uniform: bursts ~ U{1,5}, counts ~ U{2,4}, seeds 10K-10.5K",
    "Realistic: exponential bursts (Coffman-Kleinrock model)",
], font_size=18)
add_sub_bullet(tf, "SYSTEM: mu_cpu=2.5, mu_io=2.0")
add_sub_bullet(tf, "INTERACTIVE: mu_cpu=4.0, mu_io=5.0")
add_sub_bullet(tf, "BACKGROUND: mu_cpu=8.0, mu_io=2.0")
add_sub_bullet(tf, "Type mix: 20/50/30%, seeds 20K-20.5K")

# Right column — baselines & training
add_text_box(slide, Inches(6.8), Inches(1.3), Inches(5.5), Inches(0.5),
             "Baselines (9 schedulers)", font_size=24, bold=True, color=ACCENT_BLUE)
txBox2 = slide.shapes.add_textbox(Inches(6.8), Inches(1.9), Inches(5.5), Inches(2.5))
tf2 = txBox2.text_frame
tf2.word_wrap = True
add_bullet_slide_content(tf2, [
    "Classical: FCFS, Round Robin (q=2), Priority+Aging, MLFQ (2/4/8), CFS, Stride",
    "RL ablation: Plain MaskablePPO (no graph)",
    "Oracle: SRTF (uses hidden future bursts)",
], font_size=18)

# Training box
box = add_rect(slide, Inches(0.7), Inches(4.8), Inches(11.8), Inches(2.0),
               LIGHT_GRAY, RGBColor(0xD5, 0xD8, 0xDC))
add_text_box(slide, Inches(1.0), Inches(4.95), Inches(3.0), Inches(0.5),
             "Training & Evaluation", font_size=22, bold=True, color=DARK_BLUE)
add_text_box(slide, Inches(1.0), Inches(5.5), Inches(5.0), Inches(1.2),
             "150,000 timesteps | lr = 3e-4 | n_steps = 2048 | clip = 0.2\n500 paired episodes per condition | Bootstrap 95% CI (5,000 resamples)\nMulti-seed: 5 training seeds x 500 test workloads",
             font_size=17, color=DARK_GRAY)

# ============================================================
# SLIDE 7: RESULTS — AVERAGE WAITING TIME (with chart)
# ============================================================
slide = make_content_slide("Results: Average Waiting Time")

slide.shapes.add_picture(
    os.path.join(FIGURES, "fig2a_avg_wait.png"),
    Inches(0.5), Inches(1.2), Inches(12.3))

# Key number callouts at bottom
add_text_box(slide, Inches(0.7), Inches(6.2), Inches(5.5), Inches(0.8),
             "Uniform: 18.54 vs 24.31 (CFS) = 23.7% reduction",
             font_size=20, bold=True, color=ACCENT_GREEN)
add_text_box(slide, Inches(6.8), Inches(6.2), Inches(5.5), Inches(0.8),
             "Realistic: 28.00 vs 28.33 (CFS) = ~1.2% (effectively a tie)",
             font_size=20, bold=True, color=ACCENT_ORANGE)

# ============================================================
# SLIDE 8: RESULTS — STARVATION (with chart)
# ============================================================
slide = make_content_slide("Results: Starvation Rate")

slide.shapes.add_picture(
    os.path.join(FIGURES, "fig2b_starvation.png"),
    Inches(0.5), Inches(1.2), Inches(12.3))

add_text_box(slide, Inches(0.7), Inches(6.2), Inches(5.5), Inches(0.8),
             "Uniform: 62.9% vs 87.3% (CFS) = -24.4 pp",
             font_size=20, bold=True, color=ACCENT_GREEN)
add_text_box(slide, Inches(6.8), Inches(6.2), Inches(5.5), Inches(0.8),
             "Realistic: 67.7% vs 79.2% (CFS) = -11.5 pp",
             font_size=20, bold=True, color=ACCENT_GREEN)

# ============================================================
# SLIDE 9: FULL BENCHMARK TABLE
# ============================================================
slide = make_content_slide("Full Benchmark (Table I)")

# Recreate Table I as a visual table
rows_data = [
    ("Scheduler", "Uniform\nAvg Wait", "Uniform\nStarv.", "Realistic\nAvg Wait", "Realistic\nStarv."),
    ("FCFS", "23.91", "88.5%", "34.56", "88.4%"),
    ("Round Robin", "24.70", "89.1%", "30.93", "86.5%"),
    ("Priority+Aging", "24.78", "90.4%", "31.79", "87.9%"),
    ("MLFQ", "24.85", "90.2%", "31.76", "87.5%"),
    ("CFS", "24.31", "87.3%", "28.33", "79.2%"),
    ("Stride", "24.30", "87.3%", "28.35", "79.3%"),
    ("Plain MaskablePPO", "19.30", "65.4%", "28.28", "68.5%"),
    ("GS + MaskablePPO", "18.54", "62.9%", "28.00", "67.7%"),
    ("SRTF (oracle)", "14.49", "53.1%", "17.32", "52.4%"),
]

table_shape = slide.shapes.add_table(len(rows_data), 5,
                                      Inches(1.5), Inches(1.3),
                                      Inches(10.0), Inches(5.5))
table = table_shape.table

# Style header row
for col_idx in range(5):
    cell = table.cell(0, col_idx)
    cell.text = rows_data[0][col_idx]
    for p in cell.text_frame.paragraphs:
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.font.name = "Calibri"
        p.alignment = PP_ALIGN.CENTER
    cell.fill.solid()
    cell.fill.fore_color.rgb = DARK_BLUE

# Fill data rows
for row_idx in range(1, len(rows_data)):
    for col_idx in range(5):
        cell = table.cell(row_idx, col_idx)
        cell.text = rows_data[row_idx][col_idx]
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(14)
            p.font.name = "Calibri"
            p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT
            # Highlight our model row
            if row_idx == 8:
                p.font.bold = True
                p.font.color.rgb = ACCENT_GREEN
            elif row_idx == 9:
                p.font.color.rgb = MED_GRAY
            else:
                p.font.color.rgb = DARK_GRAY
        # Alternate row coloring
        if row_idx == 8:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xE8, 0xF8, 0xF5)
        elif row_idx % 2 == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT_GRAY

# ============================================================
# SLIDE 10: CONTRIBUTION DECOMPOSITION
# ============================================================
slide = make_content_slide("Contribution Decomposition")

# Left: chart
slide.shapes.add_picture(
    os.path.join(FIGURES, "fig5_decomposition.png"),
    Inches(0.3), Inches(1.2), Inches(7.0))

# Right: key findings
add_text_box(slide, Inches(7.5), Inches(1.3), Inches(5.2), Inches(0.5),
             "Uniform Workloads", font_size=22, bold=True, color=ACCENT_BLUE)
txBox = slide.shapes.add_textbox(Inches(7.5), Inches(1.9), Inches(5.2), Inches(2.0))
tf = txBox.text_frame
tf.word_wrap = True
add_bullet_slide_content(tf, [
    "RL closes 49% of gap to SRTF oracle",
    "Graph adds another 8%",
    "Both statistically significant (CI excludes 0)",
], font_size=18)

add_text_box(slide, Inches(7.5), Inches(4.0), Inches(5.2), Inches(0.5),
             "Realistic Workloads", font_size=22, bold=True, color=ACCENT_ORANGE)
txBox2 = slide.shapes.add_textbox(Inches(7.5), Inches(4.6), Inches(5.2), Inches(2.5))
tf2 = txBox2.text_frame
tf2.word_wrap = True
add_bullet_slide_content(tf2, [
    "RL alone is WORSE than CFS (+0.6, n.s.)",
    "With graph: -0.7 (significant!)",
    "The graph is what makes RL viable",
], font_size=18)
add_sub_bullet(tf2, "Remaining gap to SRTF: 10.9 ticks (information-theoretic)")

# ============================================================
# SLIDE 11: MULTI-SEED ROBUSTNESS
# ============================================================
slide = make_content_slide("Multi-Seed Robustness")

# Simple table
ms_data = [
    ("Model", "Avg Wait", "95% CI"),
    ("Plain MaskablePPO", "21.62", "[21.20, 22.06]"),
    ("GraphSAGE + MaskablePPO", "20.96", "[20.86, 21.07]"),
    ("Paired diff (G - P)", "-0.67", "[-1.12, -0.26]"),
]

table_shape = slide.shapes.add_table(4, 3, Inches(1.5), Inches(1.5),
                                      Inches(8.0), Inches(2.5))
table = table_shape.table
for row_idx in range(4):
    for col_idx in range(3):
        cell = table.cell(row_idx, col_idx)
        cell.text = ms_data[row_idx][col_idx]
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(18)
            p.font.name = "Calibri"
            p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT
            if row_idx == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
            elif row_idx == 3:
                p.font.bold = True
                p.font.color.rgb = ACCENT_GREEN
            else:
                p.font.color.rgb = DARK_GRAY
        if row_idx == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = DARK_BLUE

# Key takeaway
box = add_rect(slide, Inches(1.5), Inches(4.5), Inches(10.0), Inches(2.0),
               RGBColor(0xE8, 0xF8, 0xF5), ACCENT_GREEN)
add_text_box(slide, Inches(1.8), Inches(4.7), Inches(9.4), Inches(0.5),
             "Key Takeaway", font_size=22, bold=True, color=ACCENT_GREEN)
add_text_box(slide, Inches(1.8), Inches(5.3), Inches(9.4), Inches(1.0),
             "5 independently trained models, each tested on 500 workloads.\nGraphSAGE advantage (-0.67 ticks) is small but highly consistent across all seeds.\n95% CI excludes zero — the improvement is real, not a lucky seed.",
             font_size=18, color=DARK_GRAY)

# ============================================================
# SLIDE 12: DASHBOARD DEMO
# ============================================================
slide = make_content_slide("Interactive Dashboard (Streamlit)")

# Two screenshots side by side
slide.shapes.add_picture(
    os.path.join(FIGURES, "fig3_dashboard_state.png"),
    Inches(0.5), Inches(1.3), Inches(6.0))

slide.shapes.add_picture(
    os.path.join(FIGURES, "fig4_process_graph.png"),
    Inches(6.8), Inches(1.3), Inches(5.8))

# Captions
add_text_box(slide, Inches(0.5), Inches(5.8), Inches(6.0), Inches(1.2),
             "Per-process state table + Gantt timeline\nP2 (SYSTEM) accumulates highest wait due to I/O;\nP4 (BACKGROUND) gets priority after aging threshold",
             font_size=15, color=MED_GRAY, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(6.8), Inches(5.5), Inches(5.8), Inches(1.2),
             "Dynamic process graph with 3 edge families:\nCPU contention (solid), type similarity (dashed),\nburst-history similarity (dotted)",
             font_size=15, color=MED_GRAY, alignment=PP_ALIGN.CENTER)

# ============================================================
# SLIDE 13: KEY TAKEAWAYS
# ============================================================
slide = make_content_slide("Key Takeaways")

takeaways = [
    ("1", "Starvation drops 12-24 pp vs CFS",
     "The learned aging reward works. Across every condition, fewer processes starve.", ACCENT_GREEN),
    ("2", "The graph is NOT optional",
     "On realistic workloads, a flat RL agent cannot outperform CFS. The relational structure provides the signal.", ACCENT_BLUE),
    ("3", "23.7% wait reduction on uniform workloads",
     "Best non-oracle performance. Statistically significant with 500 paired episodes.", ACCENT_ORANGE),
    ("4", "CFS is surprisingly hard to beat on realistic workloads",
     "Virtual-runtime weighting handles heterogeneous exponential bursts well. The gap to SRTF (10.9 ticks) is mostly information-theoretic.", DARK_BLUE),
]

for i, (num, title, body, color) in enumerate(takeaways):
    top = Inches(1.3 + i * 1.45)
    # Number circle
    circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.7), top + Inches(0.05),
                                     Inches(0.6), Inches(0.6))
    circle.fill.solid()
    circle.fill.fore_color.rgb = color
    circle.line.fill.background()
    tf = circle.text_frame
    tf.paragraphs[0].text = num
    tf.paragraphs[0].font.size = Pt(22)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = WHITE
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    add_text_box(slide, Inches(1.6), top, Inches(10.5), Inches(0.45),
                 title, font_size=22, bold=True, color=color)
    add_text_box(slide, Inches(1.6), top + Inches(0.45), Inches(10.5), Inches(0.8),
                 body, font_size=17, color=DARK_GRAY)

# ============================================================
# SLIDE 14: LIMITATIONS & FUTURE WORK
# ============================================================
slide = make_content_slide("Limitations & Future Work")

# Left: Limitations
add_text_box(slide, Inches(0.7), Inches(1.3), Inches(5.5), Inches(0.5),
             "Limitations", font_size=24, bold=True, color=ACCENT_ORANGE)
txBox = slide.shapes.add_textbox(Inches(0.7), Inches(1.9), Inches(5.5), Inches(4.5))
tf = txBox.text_frame
tf.word_wrap = True
add_bullet_slide_content(tf, [
    "Python simulator only — no kernel integration yet",
    "Fixed observation: max 8 processes",
    "\"Realistic\" workloads from textbook parameters,\nnot production /proc traces",
    "All 3 edge types share same mean-pooling;\nrelation-specific kernels (R-GCN) may help",
], font_size=19)

# Right: Future Work
add_text_box(slide, Inches(6.8), Inches(1.3), Inches(5.5), Inches(0.5),
             "Future Work", font_size=24, bold=True, color=ACCENT_GREEN)
txBox2 = slide.shapes.add_textbox(Inches(6.8), Inches(1.9), Inches(5.5), Inches(4.5))
tf2 = txBox2.text_frame
tf2.word_wrap = True
add_bullet_slide_content(tf2, [
    "eBPF integration into real Linux kernel",
    "Variable-length process queues with\nsparse graph attention",
    "Evaluation on production workload traces",
    "R-GCN for relation-specific edge kernels",
], font_size=19)

# ============================================================
# SLIDE 15: THANK YOU
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_rect(slide, Inches(0), Inches(0), W, Inches(4.2), DARK_BLUE)
add_rect(slide, Inches(0), Inches(4.2), W, Inches(0.08), ACCENT_BLUE)

add_text_box(slide, Inches(0.8), Inches(1.2), Inches(11.5), Inches(1.0),
             "Thank You", font_size=48, bold=True, color=WHITE,
             alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(0.8), Inches(2.3), Inches(11.5), Inches(0.6),
             "Questions?", font_size=28, color=RGBColor(0xAE, 0xD6, 0xF1),
             alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(0.8), Inches(3.2), Inches(11.5), Inches(0.6),
             "V. Saadhanaa  |  Disha Vishal Chaudhari  |  Sankari M.",
             font_size=18, color=WHITE, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(0.8), Inches(5.0), Inches(11.5), Inches(1.5),
             "saadhanaa.2025@vitstudent.ac.in\ndisha.vishal2025@vitstudent.ac.in\nsankari.m@vit.ac.in",
             font_size=16, color=MED_GRAY, alignment=PP_ALIGN.CENTER)

# ============================================================
# SAVE
# ============================================================
prs.save(OUT)
print(f"Saved: {OUT}")
print(f"Slides: {len(prs.slides)}")
