"""CPU Scheduling Lab: local, step-by-step Streamlit demonstration."""
import os, sys
import numpy as np
import streamlit as st

st.set_page_config(page_title="CPU Scheduling Lab", page_icon="◈", layout="wide")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from rl_env import CPUSchedulingEnv, IDLE_ACTION, MAX_PROCESSES, PROCESS_WAIT_WEIGHT
from process_graph import build_process_graph
from scheduler_baselines import FCFSPolicy, PriorityAgingPolicy, RoundRobinPolicy

TYPE_COLORS = {"SYSTEM": "#ff7b72", "INTERACTIVE": "#58a6ff", "BACKGROUND": "#d29922"}
PAPER_TYPE_COLORS = {"SYSTEM": "#9d3e35", "INTERACTIVE": "#356e9f", "BACKGROUND": "#8c6d1f"}
STATE_COLORS = {"READY": "#3fb950", "BLOCKED": "#8b949e", "FINISHED": "#484f58"}
MODEL_FILES = {"Plain MaskablePPO": "maskable_ppo_variable_scheduler", "GraphSAGE + MaskablePPO": "graphsage_maskable_ppo_variable_scheduler"}


def demo_css():
    st.markdown("""<style>
    .stApp{background:#0d1117;color:#e6edf3}.block-container{max-width:1480px;padding:2.2rem 3rem 4rem}
    .hero{border:1px solid #30363d;border-radius:18px;padding:22px 26px;background:linear-gradient(115deg,#161b22,#10151d);margin-bottom:18px}
    .eyebrow{color:#8b949e;text-transform:uppercase;letter-spacing:.16em;font-size:.72rem}.hero h1{margin:.25rem 0;font-size:2.2rem;letter-spacing:-.04em}.hero p{color:#8b949e;margin:0}
    .pill{display:inline-block;margin-top:14px;padding:4px 10px;border-radius:999px;background:#1f2937;color:#79c0ff;font-size:.76rem;border:1px solid #30363d}
    .process-card{border:1px solid #30363d;border-radius:14px;padding:14px;background:#161b22;min-height:112px}.selected{border-color:#58a6ff;box-shadow:0 0 0 1px #58a6ff inset}.blocked{opacity:.72}.muted{color:#8b949e;font-size:.8rem}.section-title{margin:1.2rem 0 .55rem;font-size:1.05rem;font-weight:650}.metric{border-left:3px solid #58a6ff;padding:3px 0 3px 12px}.metric .value{font-size:1.35rem;font-weight:650}.metric .label{color:#8b949e;font-size:.76rem}
    </style>""", unsafe_allow_html=True)


def paper_css():
    st.markdown("""<style>
    .stApp,[data-testid="stAppViewContainer"]{background:#f7f7f5!important;color:#202124!important}.block-container{max-width:1500px;padding:1.15rem 1.6rem 2.5rem}
    [data-testid="stHeader"]{background:#f7f7f5}.paper-title{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;border-bottom:1px solid #bfc3c5;padding:0 0 .55rem;margin:.1rem 0 1rem}.paper-title h1{font-size:1.2rem;margin:0;font-weight:650;letter-spacing:.01em}.paper-title p{font-size:.78rem;color:#5f6368;margin:.25rem 0 0}.paper-figure{border:1px solid #c8ccce;background:#fff;padding:.75rem .85rem;margin:.9rem 0}h2{font-family:ui-monospace,SFMono-Regular,Menlo,monospace!important;font-size:.86rem!important;margin:0 0 .7rem!important;color:#202124!important}.paper-note{font: .72rem ui-monospace,SFMono-Regular,Menlo,monospace;color:#5f6368}.paper-table{border-collapse:collapse;width:100%;font: .76rem ui-monospace,SFMono-Regular,Menlo,monospace}.paper-table th{background:#f1f2f1;text-align:left;font-weight:650}.paper-table th,.paper-table td{border:1px solid #d4d7d8;padding:.38rem .42rem}.paper-selected{background:#eef4f7}.paper-status-ready{color:#2f6e4e}.paper-status-blocked{color:#73777a}.paper-status-finished{color:#73777a}.paper-metric{border:1px solid #d4d7d8;padding:.42rem .5rem;background:#fff}.paper-metric .v{font:650 .96rem ui-monospace,SFMono-Regular,Menlo,monospace}.paper-metric .l{font-size:.67rem;color:#5f6368;margin-top:.12rem}.stButton>button{border-radius:2px!important;border:1px solid #9ea3a6!important;background:#fff!important;color:#202124!important;font-size:.78rem!important}.stButton>button[kind="primary"]{background:#2d5f86!important;color:#fff!important;border-color:#2d5f86!important}[data-testid="stRadio"] label,[data-testid="stRadio"] p,[data-testid="stWidgetLabel"] p{color:#202124!important}[data-baseweb="select"]>div,[data-testid="stNumberInput"] input{background:#fff!important;color:#202124!important;border-color:#9ea3a6!important}[data-baseweb="select"] svg{fill:#202124!important}
    </style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(label):
    from sb3_contrib import MaskablePPO
    path = os.path.join(ROOT, MODEL_FILES[label])
    return MaskablePPO.load(path) if os.path.exists(path + ".zip") else None


def new_env(seed, aging, process_count):
    env = CPUSchedulingEnv(num_processes=int(process_count), starvation_threshold=10, aging_factor=aging); env.reset(seed=int(seed)); return env


def reset_live():
    st.session_state.env = new_env(st.session_state.seed, st.session_state.aging, st.session_state.process_count)
    st.session_state.timeline, st.session_state.selected, st.session_state.last_action = [], None, None
    st.session_state.comparison = None


def action_for(env, model):
    mask = env.action_masks()
    if model is None: return int(np.flatnonzero(mask)[0])
    action, _ = model.predict(env._get_observation(), deterministic=True, action_masks=mask); return int(action)


def baseline_specs(aging):
    """Small, explainable reference schedulers used for the comparison table."""
    return [
        ("FCFS", lambda: FCFSPolicy()),
        ("Round Robin (q=2)", lambda: RoundRobinPolicy(quantum=2)),
        ("Priority + aging", lambda: PriorityAgingPolicy(aging)),
    ]


def step_live():
    env, model = st.session_state.env, st.session_state.model
    if all(p["finished"] for p in env.processes): return
    action = action_for(env, model); start = env.current_time
    label = "IDLE" if action == IDLE_ACTION else env.processes[action]["name"]
    _, reward, _, _, _ = env.step(action)
    st.session_state.timeline.append({"start": start, "end": start + 1, "label": label, "reward": reward})
    st.session_state.selected = action if action < len(env.processes) else None; st.session_state.last_action = label


def state(p): return "FINISHED" if p["finished"] else ("READY" if p["ready"] else "BLOCKED")


def metrics(env):
    waits = [p["waiting"] for p in env.processes]
    weighted_waits = [PROCESS_WAIT_WEIGHT[p["type"]] * p["waiting"] for p in env.processes]
    out = {"Average waiting": np.mean(waits), "Priority-weighted avg wait": np.mean(weighted_waits), "Current max waiting": max(waits), "Elapsed time": env.current_time,
           "Starvation count": sum(w > 10 for w in waits), "Starvation rate": sum(w > 10 for w in waits) / len(waits)}
    for typ in TYPE_COLORS:
        vals = [p["waiting"] for p in env.processes if p["type"] == typ]; out[typ + " avg wait"] = np.mean(vals) if vals else 0
    return out


def timeline_chart(history, paper=False):
    import plotly.graph_objects as go
    colors = PAPER_TYPE_COLORS if paper else TYPE_COLORS
    fig = go.Figure()
    for x in history:
        ptype = next((p["type"] for p in st.session_state.env.processes if p["name"] == x["label"]), None)
        fig.add_trace(go.Bar(x=[1], y=["CPU"], base=[x["start"]], orientation="h", name=x["label"], text=[x["label"]], textposition="inside", showlegend=False,
                             marker_color="#73777a" if not ptype else colors[ptype], hovertemplate=f"{x['label']} · t={x['start']}→{x['end']}<extra></extra>"))
    background = "#ffffff" if paper else "rgba(0,0,0,0)"
    font_color = "#202124" if paper else "#8b949e"
    fig.update_layout(height=145,barmode="stack",margin=dict(l=10,r=10,t=10,b=10),xaxis_title="scheduler time",xaxis=dict(tickfont=dict(color=font_color),title_font=dict(color=font_color)),yaxis=dict(showticklabels=False),paper_bgcolor=background,plot_bgcolor=background,font=dict(color=font_color),showlegend=False)
    return fig


def graph_chart(env, selected, paper=False):
    import plotly.graph_objects as go
    colors = PAPER_TYPE_COLORS if paper else TYPE_COLORS
    graph = build_process_graph(env); n = len(env.processes); angles = np.linspace(0, 2*np.pi, n, endpoint=False); pos = {i:(np.cos(a),np.sin(a)) for i,a in enumerate(angles)}
    fig = go.Figure()
    for relation, dash in (("contention", "solid"), ("type", "dash"), ("history", "dot")):
        ex, ey = [], []
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = env.processes[i], env.processes[j]
                active = relation == "contention" and pi["ready"] and pj["ready"] and not pi["finished"] and not pj["finished"]
                active |= relation == "type" and pi["type"] == pj["type"]
                active |= relation == "history" and pi["average_burst"] > 0 and pj["average_burst"] > 0 and abs(pi["average_burst"] - pj["average_burst"]) <= 1
                if active: ex += [pos[i][0], pos[j][0], None]; ey += [pos[i][1], pos[j][1], None]
        fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",name=relation,line=dict(color="#6e7681",dash=dash),hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[pos[i][0] for i in range(n)],y=[pos[i][1] for i in range(n)],mode="markers+text",
        text=[f"{p['name']}<br>{p['type'][:3]} · W{p['waiting']}" for p in env.processes],textposition="bottom center",
        marker=dict(size=34,color=[colors[p["type"]] for p in env.processes],symbol=["circle" if state(p)=="READY" else ("diamond" if state(p)=="BLOCKED" else "x") for p in env.processes],opacity=[1 if state(p)=="READY" else .55 for p in env.processes],line=dict(width=[3 if i==selected else 1 for i in range(n)],color="#202124" if paper else "#f0f6fc")),hovertext=[f"{p['name']} · {state(p)}" for p in env.processes],hoverinfo="text",name="processes"))
    background = "#ffffff" if paper else "rgba(0,0,0,0)"
    font_color = "#202124" if paper else "#8b949e"
    fig.update_layout(height=400,margin=dict(l=10,r=10,t=10,b=10),xaxis=dict(visible=False),yaxis=dict(visible=False),paper_bgcolor=background,plot_bgcolor=background,font=dict(color=font_color),legend=dict(orientation="h",y=-.05,font=dict(color=font_color)))
    return fig


def process_cards(env):
    for start in range(0, len(env.processes), 4):
        row = env.processes[start:start + 4]
        for col, (i, p) in zip(st.columns(len(row)), enumerate(row, start)):
            s = state(p); selected = st.session_state.selected == i
            col.markdown(f"<div class='process-card {'selected' if selected else ''} {'blocked' if s != 'READY' else ''}'><b>{p['name']}</b> <span style='float:right;color:{TYPE_COLORS[p['type']]}'>{p['type']}</span><div style='margin:10px 0;color:{STATE_COLORS[s]}'><b>{s}</b>{' · RUNNING' if selected else ''}</div><div class='muted'>wait <b>{p['waiting']}</b> · CPU <b>{p['cpu_used']}</b> · scheduled <b>{p['times_scheduled']}</b></div><div class='muted'>last burst {p['last_burst']} · avg history {p['average_burst']:.1f} · blocked {p['blocked_time']}</div></div>", unsafe_allow_html=True)


def compare(seed, aging, process_count):
    rows=[]
    schedulers = baseline_specs(aging) + [(label, lambda label=label: load_model(label)) for label in MODEL_FILES]
    for label, make_scheduler in schedulers:
        scheduler = make_scheduler()
        if scheduler is None: continue
        env=new_env(seed,aging,process_count)
        while not all(p["finished"] for p in env.processes):
            action = scheduler.act(env) if hasattr(scheduler, "act") else action_for(env, scheduler)
            env.step(action)
        rows.append({"Model":label,**metrics(env)})
    return rows


def render_demo_mode():
    st.markdown("<div class='hero'><div class='eyebrow'>OS + RL + Graph Engineering</div><h1>CPU Scheduling Lab</h1><p>Dynamic process scheduling using MaskablePPO and GraphSAGE</p><span class='pill'>STEP-MODE RESEARCH DEMO</span></div>",unsafe_allow_html=True)
    with st.container(border=True):
        c1,c2,c3,c4,c5,c6=st.columns([2.3,1,1,1,1,1.25])
        with c1: selected=st.selectbox("Scheduler",list(MODEL_FILES),index=list(MODEL_FILES).index(st.session_state.model_name))
        with c2: seed=st.number_input("Seed",0,999999,st.session_state.seed)
        with c3: aging=st.number_input("Aging factor",0.,5.,.5,step=.25)
        with c4: process_count=st.number_input("Processes",1,MAX_PROCESSES,st.session_state.process_count)
        with c5:
            if st.button("Generate",use_container_width=True): st.session_state.seed=int(seed);st.session_state.aging=float(aging);st.session_state.process_count=int(process_count);st.session_state.model_name=selected;st.session_state.model=load_model(selected);reset_live();st.rerun()
        with c6:
            if st.button("Reset",use_container_width=True): reset_live();st.rerun()
        if st.session_state.model is None: st.warning(f"Checkpoint missing: {MODEL_FILES[st.session_state.model_name]}.zip")
    env=st.session_state.env; st.markdown("<div class='section-title'>Legend</div>",unsafe_allow_html=True); st.markdown(" · ".join([f"<span style='color:{TYPE_COLORS[t]}'><b>● {t}</b></span>" for t in TYPE_COLORS] + ["<span style='color:#8b949e'><b>● IDLE</b></span>"]),unsafe_allow_html=True); st.markdown("<div class='section-title'>Live process state</div>",unsafe_allow_html=True); process_cards(env)
    st.markdown("<div class='section-title'>CPU execution timeline</div>",unsafe_allow_html=True); st.plotly_chart(timeline_chart(st.session_state.timeline),use_container_width=True,config={"displayModeBar":False})
    c1,c2=st.columns([1.1,1.8])
    with c1:
        st.markdown("<div class='section-title'>Step controls</div>",unsafe_allow_html=True); a,b,c=st.columns(3)
        with a:
            if st.button("Step",type="primary",use_container_width=True): step_live();st.rerun()
        with b:
            if st.button("Run episode",use_container_width=True):
                while not all(p["finished"] for p in env.processes): step_live()
                st.rerun()
        with c:
            if st.button("Compare",use_container_width=True): st.session_state.comparison=compare(st.session_state.seed,st.session_state.aging,st.session_state.process_count)
        ready=[p["name"] for p in env.processes if state(p)=="READY"]; blocked=[p["name"] for p in env.processes if state(p)=="BLOCKED"]
        st.markdown(f"**READY QUEUE**  {' → '.join(ready) if ready else '—'}"); st.markdown(f"**BLOCKED**  {', '.join(blocked) if blocked else '—'}"); st.markdown(f"**Action mask**  `{env.action_masks().astype(int).tolist()}`"); st.markdown(f"**Last action**  `{st.session_state.last_action or '—'}`")
    with c2:
        st.markdown("<div class='section-title'>Dynamic process graph</div>",unsafe_allow_html=True); st.plotly_chart(graph_chart(env,st.session_state.selected),use_container_width=True,config={"displayModeBar":False})
    st.markdown("<div class='section-title'>Episode metrics</div>",unsafe_allow_html=True); ms=metrics(env)
    for col,key in zip(st.columns(len(ms)),ms):
        value = f"{ms[key]:.1%}" if key == "Starvation rate" else f"{ms[key]:.2f}"
        col.markdown(f"<div class='metric'><div class='value'>{value}</div><div class='label'>{key}</div></div>",unsafe_allow_html=True)
    with st.expander("Decision Context"): st.write({"chosen_action":st.session_state.last_action or "—","legal_action_mask":env.action_masks().tolist()})
    with st.expander("Research Debug View — hidden workload values"): st.caption("Hidden from RL policy — visible only for researcher inspection.");st.json([{p["name"]:{"cpu_bursts":p["cpu_bursts"],"io_bursts":p["io_bursts"]}} for p in env.processes])
    if st.session_state.get("comparison"):
        import pandas as pd, plotly.express as px
        st.markdown("<div class='section-title'>Baseline comparison · same workload / same seed</div>",unsafe_allow_html=True);df=pd.DataFrame(st.session_state.comparison);st.plotly_chart(px.bar(df,x="Model",y="Average waiting",color="Model",template="plotly_dark"),use_container_width=True);st.dataframe(df,use_container_width=True,hide_index=True)
    summary = os.path.join(ROOT, "results", "summary_results.csv")
    if os.path.exists(summary):
        import pandas as pd
        benchmark = pd.read_csv(summary)
        with st.expander("Verified 500-workload benchmark"):
            st.caption("500 identical seeded workloads · aging factor = 0.5")
            main_columns = ["model", "priority_weighted_waiting", "average_waiting", "mean_episode_max_waiting", "completion_time", "starvation_rate"]
            st.dataframe(benchmark[main_columns], use_container_width=True, hide_index=True)
        with st.expander("Detailed benchmark metrics"):
            detail_columns = ["model", "global_max_waiting", "p95_waiting", "p99_waiting", "SYSTEM_waiting", "INTERACTIVE_waiting", "BACKGROUND_waiting"]
            st.dataframe(benchmark[detail_columns], use_container_width=True, hide_index=True)
    multi_seed = os.path.join(ROOT,"results","multi_seed_summary.csv")
    if os.path.exists(multi_seed):
        import pandas as pd
        with st.expander("Five independently trained seeds · RL stability check"): st.dataframe(pd.read_csv(multi_seed),use_container_width=True,hide_index=True)
    paired_seed = os.path.join(ROOT,"results","multi_seed_paired_average_waiting.csv")
    if os.path.exists(paired_seed):
        import pandas as pd
        st.caption("Five-seed paired average-waiting test")
        st.dataframe(pd.read_csv(paired_seed),use_container_width=True,hide_index=True)
    versus_fcfs = os.path.join(ROOT,"results","multi_seed_graph_vs_fcfs.csv")
    if os.path.exists(versus_fcfs):
        import pandas as pd
        st.caption("Five-seed Graph-RL versus FCFS")
        st.dataframe(pd.read_csv(versus_fcfs),use_container_width=True,hide_index=True)
    st.markdown("<div class='section-title'>Architecture</div>",unsafe_allow_html=True);st.code("Observable state → Dynamic process graph → GraphSAGE → MaskablePPO → Legal action → CPU / I/O environment\nObservable state → MaskablePPO (plain baseline)")


def paper_process_table(env):
    rows = []
    for index, process in enumerate(env.processes):
        status = state(process)
        selected = " paper-selected" if st.session_state.selected == index else ""
        rows.append(f"<tr class='{selected.strip()}'><td>{process['name']}</td><td style='color:{PAPER_TYPE_COLORS[process['type']]}'>{process['type']}</td><td class='paper-status-{status.lower()}'>{status}</td><td>{process['waiting']}</td><td>{process['cpu_used']}</td><td>{process['average_burst']:.1f}</td></tr>")
    return "<table class='paper-table'><thead><tr><th>PID</th><th>TYPE</th><th>STATE</th><th>WAIT</th><th>CPU USED</th><th>AVG BURST</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def render_paper_controls():
    c1, c2, c3, c4, c5, c6, c7 = st.columns([2.1, 1, 1, 1, .9, 1.1, .9])
    with c1:
        selected = st.selectbox("Scheduler", list(MODEL_FILES), index=list(MODEL_FILES).index(st.session_state.model_name), key="paper_scheduler")
    with c2: seed = st.number_input("Seed", 0, 999999, st.session_state.seed, key="paper_seed")
    with c3: process_count = st.number_input("Processes", 1, MAX_PROCESSES, st.session_state.process_count, key="paper_processes")
    with c4:
        if st.button("Generate", key="paper_generate", use_container_width=True):
            st.session_state.seed, st.session_state.process_count = int(seed), int(process_count)
            st.session_state.model_name, st.session_state.model = selected, load_model(selected)
            reset_live(); st.rerun()
    with c5:
        if st.button("Step", key="paper_step", type="primary", use_container_width=True): step_live(); st.rerun()
    with c6:
        if st.button("Run Episode", key="paper_run", use_container_width=True):
            while not all(process["finished"] for process in st.session_state.env.processes): step_live()
            st.rerun()
    with c7:
        if st.button("Reset", key="paper_reset", use_container_width=True): reset_live(); st.rerun()


def render_paper_figure_state(env):
    with st.container(border=True):
        st.markdown("<h2>FIGURE 1. SCHEDULING STATE</h2>", unsafe_allow_html=True)
        st.markdown(" · ".join(f"<span style='color:{PAPER_TYPE_COLORS[k]}'><b>● {k}</b></span>" for k in PAPER_TYPE_COLORS), unsafe_allow_html=True)
        left, right = st.columns([1.25, 1])
        with left: st.markdown(paper_process_table(env), unsafe_allow_html=True)
        with right:
            st.plotly_chart(timeline_chart(st.session_state.timeline, paper=True), use_container_width=True, config={"displayModeBar": False})
            cols = st.columns(3)
            current = st.session_state.last_action or "—"
            for col, (label, value) in zip(cols, (("CURRENT ACTION", current), ("TIME", env.current_time), ("READY", sum(p["ready"] and not p["finished"] for p in env.processes)))):
                col.markdown(f"<div class='paper-metric'><div class='v'>{value}</div><div class='l'>{label}</div></div>", unsafe_allow_html=True)


def render_paper_figure_graph(env):
    with st.container(border=True):
        st.markdown("<h2>FIGURE 2. DYNAMIC PROCESS GRAPH</h2>", unsafe_allow_html=True)
        left, right = st.columns([1.7, .8])
        with left: st.plotly_chart(graph_chart(env, st.session_state.selected, paper=True), use_container_width=True, config={"displayModeBar": False})
        with right:
            st.markdown("<div class='paper-note'><b>EDGE LEGEND</b><br>— CPU contention<br>-- same process type<br>·· similar burst history</div>", unsafe_allow_html=True)
            st.markdown("<br><div class='paper-note'><b>DECISION CONTEXT</b></div>", unsafe_allow_html=True)
            st.code(f"action: {st.session_state.last_action or '—'}\nmask: {env.action_masks().astype(int).tolist()}\nready: {[p['name'] for p in env.processes if p['ready'] and not p['finished']]}", language=None)


def render_paper_figure_comparison():
    import pandas as pd
    import plotly.graph_objects as go
    summary = os.path.join(ROOT, "results", "summary_results.csv")
    with st.container(border=True):
        st.markdown("<h2>FIGURE 3. MODEL COMPARISON · 500 IDENTICAL WORKLOADS</h2>", unsafe_allow_html=True)
        if not os.path.exists(summary):
            st.info("Benchmark summary is unavailable. Run research_benchmark.py to generate Figure 3.")
            return
        data = pd.read_csv(summary)
        data = data[data["model"].isin(["Plain MaskablePPO", "GraphSAGE + MaskablePPO"])].set_index("model")
        if len(data) != 2:
            st.warning("Both RL benchmark rows are required for Figure 3.")
            return
        plain, graph = data.loc["Plain MaskablePPO"], data.loc["GraphSAGE + MaskablePPO"]
        headline = [("Average waiting", "average_waiting", 1), ("Mean episode max", "mean_episode_max_waiting", 1), ("Completion time", "completion_time", 1), ("Starvation rate (%)", "starvation_rate", 100)]
        types = [("SYSTEM", "SYSTEM_waiting"), ("INTERACTIVE", "INTERACTIVE_waiting"), ("BACKGROUND", "BACKGROUND_waiting")]
        c1, c2 = st.columns(2)
        with c1:
            figure = go.Figure([go.Bar(name="Plain MaskablePPO", x=[x[0] for x in headline], y=[plain[x[1]] * x[2] for x in headline], marker_color="#777777"), go.Bar(name="GraphSAGE + MaskablePPO", x=[x[0] for x in headline], y=[graph[x[1]] * x[2] for x in headline], marker_color="#356e9f")])
            figure.update_layout(barmode="group", height=285, margin=dict(l=35,r=10,t=20,b=65), paper_bgcolor="#fff", plot_bgcolor="#fff", font=dict(color="#202124"), legend=dict(orientation="h", y=1.15, font=dict(color="#202124")), xaxis=dict(tickfont=dict(color="#202124"), title_font=dict(color="#202124")), yaxis=dict(title="Metric value", tickfont=dict(color="#202124"), title_font=dict(color="#202124")))
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
        with c2:
            figure = go.Figure([go.Bar(name="Plain MaskablePPO", x=[x[0] for x in types], y=[plain[x[1]] for x in types], marker_color="#777777"), go.Bar(name="GraphSAGE + MaskablePPO", x=[x[0] for x in types], y=[graph[x[1]] for x in types], marker_color="#356e9f")])
            figure.update_layout(barmode="group", height=285, margin=dict(l=35,r=10,t=20,b=45), paper_bgcolor="#fff", plot_bgcolor="#fff", font=dict(color="#202124"), legend=dict(orientation="h", y=1.15, font=dict(color="#202124")), xaxis=dict(tickfont=dict(color="#202124"), title_font=dict(color="#202124")), yaxis=dict(title="Average waiting", tickfont=dict(color="#202124"), title_font=dict(color="#202124")))
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def render_paper_mode():
    st.markdown("<div class='paper-title'><h1>CPU Scheduling Lab · Paper Figure View</h1><p>Reproducible scheduling-state and benchmark figures · aging factor 0.5</p></div>", unsafe_allow_html=True)
    render_paper_controls()
    env = st.session_state.env
    render_paper_figure_state(env)
    render_paper_figure_graph(env)
    render_paper_figure_comparison()
    with st.expander("Research debug: hidden from RL policy"):
        st.json([{p["name"]: {"cpu_bursts": p["cpu_bursts"], "io_bursts": p["io_bursts"]}} for p in env.processes])


def initialize_state():
    if "env" not in st.session_state:
        st.session_state.seed, st.session_state.aging, st.session_state.process_count = 42, .5, 3
        st.session_state.model_name = "GraphSAGE + MaskablePPO"
        st.session_state.model = load_model(st.session_state.model_name)
        reset_live()


def main():
    initialize_state()
    current = st.session_state.get("view_mode", "Demo Mode")
    (demo_css if current == "Demo Mode" else paper_css)()
    mode = st.radio("View Mode", ("Demo Mode", "Paper Mode"), horizontal=True, key="view_mode")
    if mode == "Demo Mode": render_demo_mode()
    else: render_paper_mode()


if __name__ == "__main__": main()
