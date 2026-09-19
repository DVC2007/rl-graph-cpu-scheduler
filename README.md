# CPU Scheduling Lab

## Running the visualization

```bash
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

- Choose Plain PPO or GraphSAGE + PPO.
- Set a seed and aging factor, then Generate.
- Use Step for exactly one scheduler decision.
- Use Run episode to complete the current workload.
- Inspect process state, queue, timeline, graph, and metrics after each step.
- Compare both checkpoints on the same seed.
- Expand Research Debug View only when inspecting hidden simulator values.
