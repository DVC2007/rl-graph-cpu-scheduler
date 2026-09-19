"""CPU Scheduling Lab: local, step-by-step Streamlit demonstration."""
import os, sys
import numpy as np
import streamlit as st

st.set_page_config(page_title="CPU Scheduling Lab", page_icon="◈", layout="wide")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from rl_env import CPUSchedulingEnv, IDLE_ACTION, MAX_PROCESSES
from process_graph import build_process_graph

TYPE_COLORS = {"SYSTEM": "#ff7b72", "INTERACTIVE": "#58a6ff", "BACKGROUND": "#d29922"}
STATE_COLORS = {"READY": "#3fb950", "BLOCKED": "#8b949e", "FINISHED": "#484f58"}
MODEL_FILES = {"Plain MaskablePPO": "maskable_ppo_variable_scheduler", "GraphSAGE + MaskablePPO": "graphsage_maskable_ppo_variable_scheduler"}


def css():
    st.markdown("""<style>
    .stApp{background:#0d1117;color:#e6edf3}.block-container{max-width:1480px;padding:2.2rem 3rem 4rem}
    .hero{border:1px solid #30363d;border-radius:18px;padding:22px 26px;background:linear-gradient(115deg,#161b22,#10151d);margin-bottom:18px}
    .eyebrow{color:#8b949e;text-transform:uppercase;letter-spacing:.16em;font-size:.72rem}.hero h1{margin:.25rem 0;font-size:2.2rem;letter-spacing:-.04em}.hero p{color:#8b949e;margin:0}
    .pill{display:inline-block;margin-top:14px;padding:4px 10px;border-radius:999px;background:#1f2937;color:#79c0ff;font-size:.76rem;border:1px solid #30363d}
    .process-card{border:1px solid #30363d;border-radius:14px;padding:14px;background:#161b22;min-height:112px}.selected{border-color:#58a6ff;box-shadow:0 0 0 1px #58a6ff inset}.blocked{opacity:.72}.muted{color:#8b949e;font-size:.8rem}.section-title{margin:1.2rem 0 .55rem;font-size:1.05rem;font-weight:650}.metric{border-left:3px solid #58a6ff;padding:3px 0 3px 12px}.metric .value{font-size:1.35rem;font-weight:650}.metric .label{color:#8b949e;font-size:.76rem}
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
    out = {"Average waiting": np.mean(waits), "Current max waiting": max(waits), "Elapsed time": env.current_time,
           "Starvation count": sum(w > 10 for w in waits), "Starvation rate": sum(w > 10 for w in waits) / len(waits)}
    for typ in TYPE_COLORS:
        vals = [p["waiting"] for p in env.processes if p["type"] == typ]; out[typ + " avg wait"] = np.mean(vals) if vals else 0
    return out


def timeline_chart(history):
    import plotly.graph_objects as go
    fig = go.Figure()
    for x in history:
        ptype = next((p["type"] for p in st.session_state.env.processes if p["name"] == x["label"]), None)
        fig.add_trace(go.Bar(x=[1], y=["CPU"], base=[x["start"]], orientation="h", name=x["label"], text=[x["label"]], textposition="inside", showlegend=False,
                             marker_color="#484f58" if not ptype else TYPE_COLORS[ptype], hovertemplate=f"{x['label']} · t={x['start']}→{x['end']}<extra></extra>"))
    fig.update_layout(height=130,barmode="stack",margin=dict(l=10,r=10,t=10,b=10),xaxis_title="scheduler time",yaxis=dict(showticklabels=False),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#8b949e")
    return fig


def graph_chart(env, selected):
    import plotly.graph_objects as go
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
        marker=dict(size=34,color=[TYPE_COLORS[p["type"]] for p in env.processes],symbol=["circle" if state(p)=="READY" else ("diamond" if state(p)=="BLOCKED" else "x") for p in env.processes],opacity=[1 if state(p)=="READY" else .55 for p in env.processes],line=dict(width=[3 if i==selected else 1 for i in range(n)],color="#f0f6fc")),hovertext=[f"{p['name']} · {state(p)}" for p in env.processes],hoverinfo="text",name="processes"))
    fig.update_layout(height=430,margin=dict(l=10,r=10,t=10,b=10),xaxis=dict(visible=False),yaxis=dict(visible=False),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",legend=dict(orientation="h",y=-.05))
    return fig


def process_cards(env):
    for start in range(0, len(env.processes), 4):
        row = env.processes[start:start + 4]
        for col, (i, p) in zip(st.columns(len(row)), enumerate(row, start)):
            s = state(p); selected = st.session_state.selected == i
            col.markdown(f"<div class='process-card {'selected' if selected else ''} {'blocked' if s != 'READY' else ''}'><b>{p['name']}</b> <span style='float:right;color:{TYPE_COLORS[p['type']]}'>{p['type']}</span><div style='margin:10px 0;color:{STATE_COLORS[s]}'><b>{s}</b>{' · RUNNING' if selected else ''}</div><div class='muted'>wait <b>{p['waiting']}</b> · CPU <b>{p['cpu_used']}</b> · scheduled <b>{p['times_scheduled']}</b></div><div class='muted'>last burst {p['last_burst']} · avg history {p['average_burst']:.1f} · blocked {p['blocked_time']}</div></div>", unsafe_allow_html=True)


def compare(seed, aging, process_count):
    rows=[]
    for label in MODEL_FILES:
        model=load_model(label)
        if model is None: continue
        env=new_env(seed,aging,process_count)
        while not all(p["finished"] for p in env.processes): env.step(action_for(env,model))
        rows.append({"Model":label,**metrics(env)})
    return rows


def main():
    css(); st.markdown("<div class='hero'><div class='eyebrow'>OS + RL + Graph Engineering</div><h1>CPU Scheduling Lab</h1><p>Dynamic process scheduling using MaskablePPO and GraphSAGE</p><span class='pill'>STEP-MODE RESEARCH DEMO</span></div>",unsafe_allow_html=True)
    if "env" not in st.session_state:
        st.session_state.seed=42; st.session_state.aging=.5; st.session_state.process_count=3; st.session_state.model_name="GraphSAGE + MaskablePPO"; st.session_state.model=load_model(st.session_state.model_name); reset_live()
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
    for col,key in zip(st.columns(8),ms):
        value = f"{ms[key]:.1%}" if key == "Starvation rate" else f"{ms[key]:.2f}"
        col.markdown(f"<div class='metric'><div class='value'>{value}</div><div class='label'>{key}</div></div>",unsafe_allow_html=True)
    with st.expander("Decision Context"): st.write({"chosen_action":st.session_state.last_action or "—","legal_action_mask":env.action_masks().tolist()})
    with st.expander("Research Debug View — hidden workload values"): st.caption("Hidden from RL policy — visible only for researcher inspection.");st.json([{p["name"]:{"cpu_bursts":p["cpu_bursts"],"io_bursts":p["io_bursts"]}} for p in env.processes])
    if st.session_state.get("comparison"):
        import pandas as pd, plotly.express as px
        st.markdown("<div class='section-title'>Plain RL vs Graph-RL · same workload / same seed</div>",unsafe_allow_html=True);df=pd.DataFrame(st.session_state.comparison);st.plotly_chart(px.bar(df,x="Model",y="Average waiting",color="Model",template="plotly_dark"),use_container_width=True);st.dataframe(df,use_container_width=True,hide_index=True)
    summary=os.path.join(ROOT,"results","summary_results.csv")
    if os.path.exists(summary):
        import pandas as pd
        with st.expander("Verified 500-workload benchmark"): st.dataframe(pd.read_csv(summary),use_container_width=True,hide_index=True)
    st.markdown("<div class='section-title'>Architecture</div>",unsafe_allow_html=True);st.code("Observable state → Dynamic process graph → GraphSAGE → MaskablePPO → Legal action → CPU / I/O environment\nObservable state → MaskablePPO (plain baseline)")


if __name__ == "__main__": main()
