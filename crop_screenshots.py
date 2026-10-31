"""Crop the actual Streamlit screenshots from the full-page debug image."""
from PIL import Image

src = "/Users/mac4dvc/rl-scheduler-project/paper/figures/debug_04_after_run.png"
out_dir = "/Users/mac4dvc/rl-scheduler-project/.claude/worktrees/xenodochial-mahavira-6bf543/paper/figures"

img = Image.open(src)
w, h = img.size
print(f"Full page: {w}x{h}")

# Figure 1: SCHEDULING STATE section — from "FIGURE 1" heading to end of metrics row
# Based on the screenshot: y starts around 260, ends around 520
fig1 = img.crop((50, 260, w - 30, 530))
fig1.save(f"{out_dir}/fig_scheduling_state.png")
print(f"Saved fig_scheduling_state.png: {fig1.size}")

# Figure 2: DYNAMIC PROCESS GRAPH section — from "FIGURE 2" heading to end of graph
# Based on the screenshot: y starts around 535, ends around 910
fig2 = img.crop((50, 535, w - 30, 920))
fig2.save(f"{out_dir}/fig_process_graph.png")
print(f"Saved fig_process_graph.png: {fig2.size}")

print("Done!")
