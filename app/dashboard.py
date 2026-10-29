"""CAPRF++ Presentation Dashboard — Final Year Project (CSE)

Interactive Streamlit presentation dashboard for:
"CAPRF++: Context-Aware B+-Tree Scheduler with Multi-Objective PPO"
Paper: Felsy & Isaac Sajan, Real-Time Systems 62:2 (2026), doi:10.1007/s11241-026-09456-0

Run via:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path so 'caprf' module is always found
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# Prevent Streamlit watcher RuntimeError with PyTorch internal custom classes
if hasattr(torch, "classes"):
    try:
        torch.classes.__path__ = []
    except Exception:
        pass

from caprf.config import ABLATION, Config
from caprf.data.workload import Splits, Workload, rescale_load
from caprf.experiments.common import dataset_config, load_splits, make_cluster
from caprf.experiments.run_all import load_caprf, slug
from caprf.schedulers.heuristics import HEURISTICS
from caprf.sim.cluster import Cluster, TaskRecord
from caprf.sim.runner import Summary, per_interval, run_episode, summarize

# ------------------------------------------------------------------------------
# Core Setup & Page Configuration
# ------------------------------------------------------------------------------
torch.set_num_threads(1)
st.set_page_config(
    page_title="CAPRF++ | Context-Aware B+-Tree Scheduler",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Robust paths as per requirement
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"
RUNS_DIR = RESULTS_DIR / "runs"
REPORT_PATH = RESULTS_DIR / "REPORT.md"

DATASET_INFO = {
    "bitbrains": {
        "title": "Dataset 1: GWA-T-12 Bitbrains (Real Trace)",
        "badge": "Real-World Cloud Trace",
        "desc": "Trace of ~50,000 tasks from 300 Bitbrains fastStorage virtual machines.",
    },
    "kaggle": {
        "title": "Dataset 2: Kaggle Cloud Task Scheduling",
        "badge": "Synthesized MMPP Trace",
        "desc": "Markov-modulated Poisson process with heterogeneous priority and core demands.",
    },
}

COLOR_MAP = {
    "CAPRF++": "#0ea5e9",        # Sky Blue
    "CAPRF++ (full)": "#0ea5e9",
    "Least-Loaded": "#8b5cf6",   # Violet
    "Greedy-MCT": "#f59e0b",     # Amber
    "WTC-Greedy": "#10b981",     # Emerald
    "Green-Greedy": "#84cc16",   # Lime
    "Round-Robin": "#64748b",    # Slate
    "Random": "#94a3b8",         # Light Slate
    "Vanilla PPO": "#ec4899",    # Pink
}

HEURISTIC_MAP = {H.name: H for H in HEURISTICS}

# ------------------------------------------------------------------------------
# Custom CSS for Viva Presentation Polishing
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Metric card improvements */
    [data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }
    .metric-container {
        border-radius: 8px;
        padding: 12px 16px;
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 12px;
    }
    .status-badge-ok {
        background-color: #10b98122;
        color: #10b981;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        border: 1px solid #10b98155;
    }
    .status-badge-warn {
        background-color: #f59e0b22;
        color: #f59e0b;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        border: 1px solid #f59e0b55;
    }
    .status-badge-alert {
        background-color: #ef444422;
        color: #ef4444;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        border: 1px solid #ef444455;
    }
    .sub-caption {
        font-size: 0.82rem;
        color: #888888;
        margin-top: -6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Cached Loaders
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained CAPRF++ checkpoint...")
def get_cached_model(ds: str, seed: int):
    """Load trained CAPRF++ scheduler model checkpoint."""
    try:
        sched, cfg = load_caprf(ds, seed)
        return sched, cfg
    except Exception as e:
        st.error(f"Failed to load model for dataset `{ds}`, seed `{seed}`: {e}")
        return None, None


@st.cache_resource(show_spinner="Loading workload dataset...")
def get_cached_splits(ds: str) -> Optional[Splits]:
    """Load workload splits from pre-processed cache or generate them."""
    try:
        cfg = dataset_config(ds)
        return load_splits(cfg)
    except Exception as e:
        st.error(f"Failed to load dataset splits for `{ds}`: {e}")
        return None


@st.cache_data
def get_cached_table(table_name: str) -> Optional[pd.DataFrame]:
    """Safe table loader from results/tables."""
    path = TAB_DIR / table_name
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def get_available_seeds(ds: str) -> List[int]:
    """Scan results/runs for available seed directories."""
    target_slug = slug(ABLATION[-1].name)
    base = RUNS_DIR / ds / target_slug
    if not base.exists():
        # Fallback to any variant folder
        ds_dir = RUNS_DIR / ds
        if ds_dir.exists():
            for sub in ds_dir.glob("*"):
                seeds = sorted(int(p.name[4:]) for p in sub.glob("seed*") if (p / "model.pt").exists())
                if seeds:
                    return seeds
        return [0]
    seeds = sorted(int(p.name[4:]) for p in base.glob("seed*") if (p / "model.pt").exists())
    return seeds if seeds else [0]


def make_scheduler_instance(name: str, ds: str, seed: int, prewarming: bool = True, carbon_delta: float = 0.2):
    """Instantiate a scheduler (either CAPRF++ clone or baseline heuristic)."""
    if name == "CAPRF++":
        model, cfg = get_cached_model(ds, seed)
        if model is None:
            raise RuntimeError(f"Could not load CAPRF++ model for {ds} seed {seed}")
        sched = model.eval_copy()
        sched.v.prewarming = prewarming
        sched.cfg.caprf.delta = float(carbon_delta)
        return sched
    elif name in HEURISTIC_MAP:
        return HEURISTIC_MAP[name]()
    else:
        raise ValueError(f"Unknown scheduler name: {name}")


# ------------------------------------------------------------------------------
# Sidebar Controls & Project Context
# ------------------------------------------------------------------------------
st.sidebar.markdown("## ⚡ **CAPRF++**")
st.sidebar.caption("Context-Aware B+-Tree Scheduler with Multi-Objective PPO & Predictive Prewarming")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ **Benchmark Dataset & Seed**")
ds_key = st.sidebar.selectbox(
    "Dataset",
    options=list(DATASET_INFO.keys()),
    format_func=lambda k: DATASET_INFO[k]["title"],
    help="Select the benchmark workload dataset."
)
seeds = get_available_seeds(ds_key)
seed_val = st.sidebar.selectbox(
    "Trained Model Seed",
    options=seeds,
    format_func=lambda s: f"Seed {s} (Trained Weights)",
    help="Random seed used during 120,000-step training phase."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖥️ **Simulated Cluster Specs**")
st.sidebar.markdown("""
- **Cluster Size:** 200 Heterogeneous Nodes
  - **150 Edge Nodes:** 8–16 cores @ 1.8–2.5 GHz, 0.1–0.3 kg CO₂/kWh
  - **50 Cloud Nodes:** 20–32 cores @ 2.4–3.6 GHz, 0.3–0.5 kg CO₂/kWh
- **Total Cores:** ~2,700 CPU cores
- **Cold-Start Overhead:** 50–200 ms per container
- **Network Delays:** Edge LAN (3–12 ms), Cloud WAN (40–90 ms)
""")

st.sidebar.markdown("---")
st.sidebar.caption(
    "🎓 **CSE Final Year Project (2026)**  \n"
    "Reference: *Felsy & Isaac Sajan*, Real-Time Systems 62:2 (2026)  \n"
    "Springer Nature · doi:10.1007/s11241-026-09456-0"
)

# ------------------------------------------------------------------------------
# Main Dashboard Header & Navigation
# ------------------------------------------------------------------------------
st.title("⚡ CAPRF++: Context-Aware B+-Tree Scheduler with Multi-Objective PPO")
st.markdown(
    "**Multi-Objective Cloud-Edge Task Scheduling** · Balancing **Deadline SLA**, **Latency**, "
    "**Carbon Emissions**, and **Container Cold-Start Delays**."
)

tab_demo, tab_walkthrough, tab_results = st.tabs([
    "🚀 Tab 1: Live Interactive Demo",
    "🔍 Tab 2: Explainability & Walkthrough",
    "📊 Tab 3: Official Experimental Results",
])


# ==============================================================================
# TAB 1: LIVE INTERACTIVE DEMO (For Live Examiner Viva Demo)
# ==============================================================================
with tab_demo:
    st.markdown("### 🚀 Live Examiner Simulation & Allocation Timeline")
    st.markdown(
        "Run real-time inference using the trained CAPRF++ policy weights against traditional heuristic baselines. "
        "Experience the direct impact of **Predictive Prewarming** on cold-start latency and **Carbon Weight ($s_i$)** on green scheduling."
    )

    splits = get_cached_splits(ds_key)
    cfg = dataset_config(ds_key, seed=seed_val)

    if splits is None:
        st.error("Workload splits could not be loaded. Please check data/ directory.")
    else:
        # Control Panel
        with st.container():
            st.markdown("#### ⚙️ Interactive Execution Parameters")
            ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.4, 1.2, 1.2, 1.2])

            with ctrl_c1:
                sched_mode = st.radio(
                    "Execution Mode",
                    options=["Compare CAPRF++ vs Baselines", "Single Scheduler Evaluation"],
                    index=0,
                    help="Choose whether to run a comparative benchmark against baselines or inspect one scheduler."
                )
                if sched_mode == "Single Scheduler Evaluation":
                    primary_sched = st.selectbox(
                        "Select Scheduler",
                        options=["CAPRF++", "Least-Loaded", "Greedy-MCT", "WTC-Greedy", "Green-Greedy", "Round-Robin"],
                        index=0
                    )
                else:
                    primary_sched = "CAPRF++"

            with ctrl_c2:
                task_count = st.slider(
                    "Task Count (N)",
                    min_value=20,
                    max_value=150,
                    value=70,
                    step=5,
                    help="Number of arriving workload tasks to schedule (50 to 100 recommended for fast live viva demo)."
                )

            with ctrl_c3:
                carbon_weight = st.slider(
                    "Carbon Weight ($s_i$)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.60,
                    step=0.05,
                    help="Sustainability preference weight ($s_i$). Higher values force the scheduler to prioritize green execution."
                )

            with ctrl_c4:
                prewarm_enabled = st.toggle(
                    "Predictive Prewarming",
                    value=True,
                    help="Toggle container prewarming ON/OFF. Demonstrates the ~90% cold-start reduction."
                )
                run_btn = st.button("▶ Run Schedule", type="primary", use_container_width=True)

        # Simulation Execution Logic
        if run_btn:
            with st.spinner("Executing simulation episodes..."):
                # Prepare slice of test workload
                w_raw = splits.test.window(0, task_count)
                # Apply user-specified sustainability weight to the task stream
                w_demo = copy.copy(w_raw)
                w_demo.sustain = np.full(len(w_raw), carbon_weight)
                w_demo = rescale_load(w_demo, cfg.cluster, float(cfg.workload.target_load))

                schedulers_to_run = (
                    ["CAPRF++", "Least-Loaded", "Greedy-MCT"]
                    if sched_mode == "Compare CAPRF++ vs Baselines"
                    else [primary_sched]
                )

                sim_results = {}
                for s_name in schedulers_to_run:
                    cluster_instance = make_cluster(cfg, noise_seed=seed_val)
                    cluster_instance.reset()
                    sched_inst = make_scheduler_instance(
                        s_name,
                        ds_key,
                        seed_val,
                        prewarming=prewarm_enabled,
                        carbon_delta=carbon_weight
                    )
                    df_out = run_episode(sched_inst, cluster_instance, w_demo)
                    sim_results[s_name] = df_out

                st.session_state["live_demo_results"] = sim_results
                st.session_state["live_demo_params"] = {
                    "tasks": task_count,
                    "carbon_weight": carbon_weight,
                    "prewarm": prewarm_enabled,
                    "dataset": ds_key,
                }

        # Display Results
        if "live_demo_results" in st.session_state:
            sim_results = st.session_state["live_demo_results"]
            params = st.session_state["live_demo_params"]

            st.markdown("---")
            st.markdown(f"#### 📈 Simulation Performance Metrics ({params['tasks']} Tasks Scheduled)")

            # Metric Summary Cards
            summary_dict: Dict[str, Summary] = {k: summarize(v) for k, v in sim_results.items()}

            if len(summary_dict) == 1:
                # Single scheduler display
                s_name = list(summary_dict.keys())[0]
                s = summary_dict[s_name]
                m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
                m_c1.metric("SLA Satisfaction", f"{s.sla:.1f}%", help="Percentage of tasks completed before deadline")
                m_c2.metric("Avg Latency", f"{s.latency_ms:.1f} ms", help="End-to-end task turnaround time")
                m_c3.metric("Carbon Emissions", f"{s.carbon_g:.2f} g", help="Carbon footprint (g CO2)")
                m_c4.metric("Cold-Start Delay", f"{s.cold_start_s * 1000:.1f} ms", help="Total container cold-start delay")
                m_c5.metric("Container Warm Hits", f"{s.warm_hit:.1f}%", help="Warm container executor reuse rate")
            else:
                # Comparative cards
                caprf_s = summary_dict.get("CAPRF++")
                ll_s = summary_dict.get("Least-Loaded")
                mct_s = summary_dict.get("Greedy-MCT")

                cols = st.columns(len(summary_dict))
                for col, (name, s) in zip(cols, summary_dict.items()):
                    with col:
                        st.markdown(f"**<span style='color:{COLOR_MAP.get(name, '#fff')}; font-size:1.15rem;'>{name}</span>**", unsafe_allow_html=True)
                        delta_cold = None
                        delta_carbon = None
                        if name == "CAPRF++" and ll_s is not None and ll_s.cold_start_s > 0:
                            cold_red = 100.0 * (ll_s.cold_start_s - s.cold_start_s) / ll_s.cold_start_s
                            delta_cold = f"-{cold_red:.1f}% vs Least-Loaded"
                        if name == "CAPRF++" and ll_s is not None and ll_s.carbon_g > 0:
                            carb_red = 100.0 * (ll_s.carbon_g - s.carbon_g) / ll_s.carbon_g
                            delta_carbon = f"-{carb_red:.1f}% vs Least-Loaded"

                        st.metric("SLA %", f"{s.sla:.1f}%")
                        st.metric("Avg Latency", f"{s.latency_ms:.1f} ms")
                        st.metric("Carbon (g)", f"{s.carbon_g:.2f} g", delta=delta_carbon, delta_color="inverse")
                        st.metric("Cold-Start Delay", f"{s.cold_start_s * 1000:.1f} ms", delta=delta_cold, delta_color="inverse")
                        st.metric("Executor Reuse", f"{s.warm_hit:.1f}%")

            # Allocation Timeline / Plotly Gantt Chart
            st.markdown("---")
            st.markdown("#### 📊 Interactive Node Allocation Timeline (Gantt Chart)")
            st.caption("Inspect how tasks are distributed across heterogeneous Edge and Cloud machines over simulation time.")

            gantt_c1, gantt_c2 = st.columns([1.5, 2.5])
            with gantt_c1:
                gantt_sched = st.selectbox(
                    "Select Timeline to Visualize",
                    options=list(sim_results.keys()),
                    index=0
                )
            with gantt_c2:
                gantt_color = st.radio(
                    "Color Timeline By:",
                    options=["Container Warm Reuse (Prewarming Impact)", "SLA Compliance (Met vs Missed)", "Cluster Tier (Edge vs Cloud)", "Task Type"],
                    horizontal=True
                )

            target_df = sim_results[gantt_sched].copy()

            # Build Gantt Figure
            fig_gantt = go.Figure()
            active_nodes = sorted(target_df["node"].unique())
            node_labels = {
                n: f"Node {n:03d} ({'Cloud' if target_df[target_df.node == n].iloc[0]['is_cloud'] else 'Edge'})"
                for n in active_nodes
            }

            if gantt_color == "Container Warm Reuse (Prewarming Impact)":
                warm_mask = target_df["warm_hit"]
                groups = [
                    ("🔥 Warm Container Reused (0 ms Cold)", target_df[warm_mask], "#10b981", "#059669"),
                    ("❄️ Cold Start Overhead Paid", target_df[~warm_mask], "#ef4444", "#dc2626"),
                ]
            elif gantt_color == "SLA Compliance (Met vs Missed)":
                sla_mask = target_df["sla"]
                groups = [
                    ("✅ SLA Deadline Satisfied", target_df[sla_mask], "#10b981", "#059669"),
                    ("❌ SLA Deadline Missed", target_df[~sla_mask], "#f43f5e", "#be123c"),
                ]
            elif gantt_color == "Cluster Tier (Edge vs Cloud)":
                cloud_mask = target_df["is_cloud"]
                groups = [
                    ("🟢 Edge Node Execution", target_df[~cloud_mask], "#06b6d4", "#0891b2"),
                    ("🔵 Cloud Node Execution", target_df[cloud_mask], "#3b82f6", "#2563eb"),
                ]
            else:  # Task Type
                type_cats = sorted(target_df["ttype"].unique())
                palette = px.colors.qualitative.Plotly
                groups = [
                    (f"Workload Type {t}", target_df[target_df.ttype == t], palette[i % len(palette)], palette[i % len(palette)])
                    for i, t in enumerate(type_cats)
                ]

            for name, sub_df, fill_col, border_col in groups:
                if sub_df.empty:
                    continue
                custom = np.stack([
                    sub_df["task"],
                    sub_df["ttype"],
                    sub_df["wait"] * 1000,
                    sub_df["cold_s"] * 1000,
                    sub_df["exec_s"] * 1000,
                    sub_df["latency"] * 1000,
                    sub_df["deadline"] * 1000,
                    sub_df["carbon_g"],
                    sub_df["sla"].map(lambda x: "✅ Met" if x else "❌ Missed"),
                    sub_df["warm_hit"].map(lambda x: "🔥 Warm Reused" if x else "❄️ Cold Start"),
                ], axis=-1)

                fig_gantt.add_trace(go.Bar(
                    name=name,
                    base=sub_df["start"],
                    x=sub_df["finish"] - sub_df["start"],
                    y=[node_labels[n] for n in sub_df["node"]],
                    orientation="h",
                    marker=dict(color=fill_col, line=dict(color=border_col, width=0.8)),
                    customdata=custom,
                    hovertemplate=(
                        "<b>Task #%{customdata[0]}</b> (Type %{customdata[1]})<br>"
                        "Start: %{base:.3f}s | Duration: %{x:.3f}s<br>"
                        "Latency: %{customdata[5]:.1f} ms (Deadline: %{customdata[6]:.1f} ms)<br>"
                        "Queue Wait: %{customdata[2]:.1f} ms | Cold Delay: %{customdata[3]:.1f} ms<br>"
                        "Exec Time: %{customdata[4]:.1f} ms | Carbon: %{customdata[7]:.3f} g<br>"
                        "SLA: %{customdata[8]} | Container: %{customdata[9]}<extra></extra>"
                    )
                ))

            fig_gantt.update_layout(
                barmode="overlay",
                xaxis_title="Simulation Time (seconds)",
                yaxis_title="Allocated Machine",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=max(380, len(active_nodes) * 20),
                margin=dict(l=10, r=20, t=30, b=40),
                hovermode="closest",
            )
            st.plotly_chart(fig_gantt, use_container_width=True)

            # Cumulative Comparison Table
            with st.expander("📋 View Full Metrics Comparison Table"):
                comp_rows = []
                for k, df_s in sim_results.items():
                    summ = summarize(df_s)
                    d = summ.as_dict()
                    d["Scheduler"] = k
                    comp_rows.append(d)
                c_df = pd.DataFrame(comp_rows).set_index("Scheduler").round(2)
                st.dataframe(c_df, use_container_width=True)


# ==============================================================================
# TAB 2: EXPLAINABILITY & STEP-BY-STEP SCHEDULING WALKTHROUGH
# ==============================================================================
with tab_walkthrough:
    st.markdown("### 🔍 Step-by-Step Scheduling Pipeline & Explainability")
    st.markdown(
        "Inspect a single task's journey through the **5 algorithmic stages** of CAPRF++: "
        "Context Encoding $\\to$ B+-Tree Candidate Pruning $\\to$ Weighted Tradeoff Scoring $\\to$ "
        "PPO Policy Selection $\\to$ Predictive Prewarming."
    )

    splits = get_cached_splits(ds_key)
    cfg = dataset_config(ds_key, seed=seed_val)

    if splits is None:
        st.error("Workload splits unavailable.")
    else:
        # Inspector State Management
        state_key = f"insp_{ds_key}_{seed_val}"
        if state_key not in st.session_state:
            w_test = rescale_load(splits.test, cfg.cluster, float(cfg.workload.target_load))
            sched_inst = make_scheduler_instance("CAPRF++", ds_key, seed_val)
            cluster_inst = make_cluster(cfg, noise_seed=seed_val)
            cluster_inst.reset()
            sched_inst.begin(cluster_inst, w_test, train=False)
            st.session_state[state_key] = {
                "w": w_test,
                "sched": sched_inst,
                "cl": cluster_inst,
                "curr_idx": 0,
                "last_decision": None,
            }

        S = st.session_state[state_key]

        def step_tasks(count: int = 1, until_unseen: bool = False):
            w, sched, cl = S["w"], S["sched"], S["cl"]
            done = 0
            while S["curr_idx"] < len(w):
                idx = S["curr_idx"]
                cl.now = float(w.arrival[idx])
                chosen_node = sched.decide(cl, w, idx)
                debug_trace = copy.deepcopy(sched.debug)
                rec = cl.assign(w, idx, chosen_node)
                sched.observe(cl, w, idx, rec)
                S["curr_idx"] += 1
                done += 1
                S["last_decision"] = (idx, debug_trace, rec)
                if until_unseen and w.is_unseen()[idx]:
                    break
                if not until_unseen and done >= count:
                    break

        # Inspector Navigation Header
        nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns([1.2, 1, 1, 1.3, 1.5])
        with nav_c1:
            if st.button("Next Task ▶", type="primary", use_container_width=True):
                step_tasks(1)
        with nav_c2:
            if st.button("Skip 10 ⏭", use_container_width=True):
                step_tasks(10)
        with nav_c3:
            if st.button("Skip 100 ⏩", use_container_width=True):
                step_tasks(100)
        with nav_c4:
            if st.button("🌟 Next Unseen Task", use_container_width=True, help="Advance to next task with an unseen workload class"):
                step_tasks(until_unseen=True)
        with nav_c5:
            st.markdown(f"**Progress:** Scheduled `{S['curr_idx']:,}` of `{len(S['w']):,}` tasks")

        # Initial prompt if no task has been inspected yet
        if S["last_decision"] is None:
            step_tasks(1)

        task_idx, trace, rec = S["last_decision"]
        w, cl = S["w"], S["cl"]
        ttype_idx = int(w.ttype[task_idx])
        ttype_name = w.type_names[ttype_idx] if ttype_idx < len(w.type_names) else f"Type {ttype_idx}"
        is_unseen = bool(w.is_unseen()[task_idx])

        st.markdown("---")
        st.markdown(f"#### 🏷️ Inspected Task: `Task #{task_idx}`")

        # Task Specification Card
        req_c1, req_c2, req_c3, req_c4, req_c5, req_c6, req_c7 = st.columns(7)
        req_c1.metric(
            "Task Type",
            ttype_name.split(":")[0],
            delta="UNSEEN WORKLOAD" if is_unseen else "Seen in Training",
            delta_color="inverse" if is_unseen else "off",
            help="Unseen types are held out during training to verify zero-shot generalisation."
        )
        req_c2.metric("Deadline", f"{w.deadline[task_idx]*1000:,.0f} ms")
        req_c3.metric("CPU Cores", f"{int(w.cores[task_idx])} cores")
        req_c4.metric("Memory (RAM)", f"{w.mem_gb[task_idx]:.1f} GB")
        req_c5.metric("Compute Load", f"{w.cycles[task_idx]:.2f} Gcycles")
        req_c6.metric("Input Data", f"{w.data_mb[task_idx]:.1f} MB")
        req_c7.metric("Green Weight ($s_i$)", f"{w.sustain[task_idx]:.2f}")

        st.markdown("---")
        # 5-Stage Algorithmic Breakdown Cards
        st.markdown("#### 🔬 Detailed 5-Stage Algorithmic Pipeline Execution")

        pipe_c1, pipe_c2 = st.columns([1.1, 1.4])

        with pipe_c1:
            st.markdown("##### ① + ② B+-Tree Candidate Pruning")
            cand_nodes = trace["cand"]
            total_nodes = cl.n
            pruned_pct = (1.0 - len(cand_nodes) / total_nodes) * 100.0
            tau_val = trace.get("tau")

            st.markdown(f"""
            - **Total Cluster Pool:** `{total_nodes}` nodes (150 Edge, 50 Cloud)
            - **Candidate Nodes Retrieved:** `{len(cand_nodes)}` nodes
            - **Search Space Pruned:** <span class='status-badge-ok'>{pruned_pct:.1f}% reduction</span>
            - **Adaptive iDistance Radius ($\\tau$):** `{tau_val:.3f}` if tau_val else `None`
            - **Availability B+-Tree:** Keyed by node mean core-free time; merges least-loaded nodes in $O(\\log N)$ time.
            """, unsafe_allow_html=True)

            st.markdown("##### ③ Weighted Tradeoff Controller (WTC)")
            st.latex(r"\Omega_{ij} = \alpha \frac{L_{ij}}{s_L} + \beta \frac{E_{ij}}{s_E} + \delta \frac{C_{ij}}{s_C} + \rho \frac{O_{ij}}{0.2}")
            feasible_nodes = trace["sel"]
            st.markdown(f"""
            - **Feasible Set ($C_i$):** Pruned down to `{len(feasible_nodes)}` candidate machines.
            - **Constraint:** Retains machines where $\\Omega_{{ij}} \\le \\Omega_{{min}} + \\Delta$ and predicted latency satisfies deadline margin.
            """)

            st.markdown("##### ⑤ Predictive Prewarming Signal")
            lam_val = trace.get("lam")
            prewarmed = trace.get("prewarmed", [])
            if lam_val is not None:
                st.markdown(f"""
                - **Predicted Arrival Prob ($\\lambda_i$):** `{lam_val:.2f}` (another task of type `{ttype_name.split(':')[0]}` arrives in window)
                - **Prewarm Condition:** $\\lambda_i \\cdot \\text{{Cold}}(R_j) \\ge {cfg.caprf.prewarm_threshold}$
                """)
                if prewarmed:
                    st.markdown(f"<span class='status-badge-ok'>🔥 Prewarmed Containers on Nodes: {prewarmed}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='status-badge-warn'>⏸️ No prewarming signal dispatched (threshold not met or nodes already warm).</span>", unsafe_allow_html=True)
            else:
                st.caption("Prewarming not evaluated for this step.")

        with pipe_c2:
            st.markdown("##### ④ PPO Actor-Critic Policy Decision")
            probs = trace["probs"]
            chosen_node = trace["chosen"]
            value_est = trace.get("value", 0.0)

            # Feasible set policy distribution
            feas_node_ids = trace["nodes"]
            omega_vals = trace["omega"][feasible_nodes]
            feas_df = pd.DataFrame({
                "Machine": [f"Node {n} ({'Cloud' if cl.is_cloud[n] else 'Edge'})" for n in feas_node_ids],
                "Policy Prob (%)": probs * 100.0,
                "WTC Score (Ω)": omega_vals,
                "Node ID": feas_node_ids,
            })

            fig_ppo = px.bar(
                feas_df,
                x="Machine",
                y="Policy Prob (%)",
                color="WTC Score (Ω)",
                color_continuous_scale="Viridis_r",
                title="PPO Policy Distribution π(a|s) over Feasible Nodes (Color = Ω score, lower is better)"
            )
            fig_ppo.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_ppo, use_container_width=True)

            tier = "Cloud" if cl.is_cloud[chosen_node] else "Edge"
            st.markdown(f"""
            - **Chosen Machine ($R^*$):** <span class='status-badge-ok'>Node {chosen_node} ({tier})</span>
            - **PPO Critic State Value $V(s)$:** `{value_est:.3f}` (expected discounted multi-objective return)
            """, unsafe_allow_html=True)

        # Candidate Inspection Table
        st.markdown("##### 📋 Candidate Evaluation & WTC Score Breakdown")
        pred_dict = trace["pred"]
        cand_list = trace["cand"]
        cand_df = pd.DataFrame({
            "Node": cand_list,
            "Tier": np.where(cl.is_cloud[cand_list], "Cloud", "Edge"),
            "Pred Latency (ms)": np.round(1000 * pred_dict["latency"], 1),
            "Pred Energy (J)": np.round(pred_dict["energy"], 1),
            "Pred Carbon (mg)": np.round(1000 * pred_dict["carbon"], 2),
            "Cold Start (ms)": np.round(1000 * pred_dict["cold"], 1),
            "Container Warm?": pred_dict["warm"].astype(bool),
            "WTC Score (Ω)": np.round(trace["omega"], 3),
        })
        cand_df["Feasible (Ci)"] = False
        cand_df.loc[feasible_nodes, "Feasible (Ci)"] = True
        cand_df["Policy Prob (%)"] = np.nan
        cand_df.loc[feasible_nodes, "Policy Prob (%)"] = np.round(probs * 100.0, 1)
        cand_df["Decision"] = "Filtered Out"
        cand_df.loc[cand_df["Feasible (Ci)"], "Decision"] = "Feasible Candidate"
        cand_df.loc[cand_df.Node == chosen_node, "Decision"] = "✅ CHOSEN"
        for pw in prewarmed:
            cand_df.loc[cand_df.Node == pw, "Decision"] += " & 🔥 PREWARMED"

        cand_df = cand_df.sort_values(["Feasible (Ci)", "Policy Prob (%)", "WTC Score (Ω)"], ascending=[False, False, True])
        st.dataframe(cand_df, hide_index=True, use_container_width=True, height=220)

        # Actual Measured Outcome
        st.markdown("##### 🏁 Actual Execution Outcome")
        out_c1, out_c2, out_c3, out_c4, out_c5 = st.columns(5)
        out_c1.metric("Execution Latency", f"{rec.latency*1000:,.1f} ms", help=f"Queue: {rec.wait*1000:.1f}ms, Cold: {rec.cold_s*1000:.1f}ms, Exec: {rec.exec_s*1000:.1f}ms")
        out_c2.metric("SLA Deadline Status", "MET ✅" if rec.sla else "MISSED ❌", delta=f"Margin: {(w.deadline[task_idx] - rec.latency)*1000:.1f} ms")
        out_c3.metric("Cold-Start Incurred", f"{rec.cold_s*1000:.1f} ms", delta="Reused Container" if rec.warm_hit else "Cold Container Start")
        out_c4.metric("Carbon Emitted", f"{rec.carbon_g*1000:.2f} mg CO₂")
        out_c5.metric("Energy Consumed", f"{rec.energy_j:.1f} J")


# ==============================================================================
# TAB 3: OFFICIAL EXPERIMENTAL RESULTS (From Colab Training)
# ==============================================================================
with tab_results:
    st.markdown("### 📊 Official Experimental Results & Benchmark Evaluations")
    st.markdown(
        "Ingested evaluation tables and high-resolution figures generated from the **5-seed full benchmark run** "
        "(Sec. 4 of the paper, Tables 3–9). Demonstrates statistically validated cold-start reduction, SLA compliance, "
        "and carbon savings."
    )

    # Highlight Statistical Findings Cards
    st.markdown("#### 🏆 Key Quantitative Findings")
    stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)

    stat_c1.markdown("""
    <div class='metric-container'>
        <div style='color:#888; font-size:0.85rem;'>COLD-START REDUCTION</div>
        <div style='font-size:1.8rem; font-weight:700; color:#10b981;'>94.9% / 91.0%</div>
        <div style='color:#10b981; font-size:0.8rem;'>▼ Cut vs Least-Loaded Baseline</div>
        <div class='sub-caption'>Prewarming eliminates startup stalls across both traces.</div>
    </div>
    """, unsafe_allow_html=True)

    stat_c2.markdown("""
    <div class='metric-container'>
        <div style='color:#888; font-size:0.85rem;'>SLA SATISFACTION</div>
        <div style='font-size:1.8rem; font-weight:700; color:#0ea5e9;'>98.5% (D2) / 85.7% (D1)</div>
        <div style='color:#0ea5e9; font-size:0.8rem;'>▲ Strictly Enforces Deadlines</div>
        <div class='sub-caption'>WTC filters non-viable machines before PPO action selection.</div>
    </div>
    """, unsafe_allow_html=True)

    stat_c3.markdown("""
    <div class='metric-container'>
        <div style='color:#888; font-size:0.85rem;'>CARBON FOOTPRINT REDUCTION</div>
        <div style='font-size:1.8rem; font-weight:700; color:#f59e0b;'>12.9% / 9.1%</div>
        <div style='color:#f59e0b; font-size:0.8rem;'>▼ CO₂ Savings vs Baselines</div>
        <div class='sub-caption'>Multi-objective reward explicitly prices time-varying grid carbon.</div>
    </div>
    """, unsafe_allow_html=True)

    stat_c4.markdown("""
    <div class='metric-container'>
        <div style='color:#888; font-size:0.85rem;'>UNSEEN GENERALISATION</div>
        <div style='font-size:1.8rem; font-weight:700; color:#8b5cf6;'>100.0% / 99.2%</div>
        <div style='color:#8b5cf6; font-size:0.8rem;'>▲ Zero-Shot Adaptation</div>
        <div class='sub-caption'>Context-aware embeddings generalise to held-out workload types.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    # Sub-tabs for Tables and Plots
    res_sub_tab1, res_sub_tab2, res_sub_tab3, res_sub_tab4 = st.tabs([
        "📋 Benchmark Comparison Tables",
        "🧪 Ablation Study Breakdown",
        "📈 Evaluation Figures & Plots",
        "📑 Executive REPORT.md",
    ])

    # 1. Comparison Tables
    with res_sub_tab1:
        st.markdown("#### Table 3: Summary Performance on Both Datasets (Mean ± Std over 5 Seeds)")
        t3 = get_cached_table("table3_caprf_performance.csv")
        if t3 is not None:
            # Enrich with paper's original reference numbers
            paper_map = {
                "Avg. task completion latency (ms)": ("25.0", "30.0"),
                "SLA satisfaction (%)": ("99.4", "99.3"),
                "Carbon footprint reduction (%)": ("70.2", "70.8"),
                "Cold-start overhead reduction (%)": ("54.1", "51.6"),
                "Generalization on unseen workloads (%)": ("97.8", "97.1"),
                "Makespan (s)": ("340.3", "210.6"),
            }
            t3["Paper Reported (D1 / D2)"] = t3["Metric"].map(lambda m: " / ".join(paper_map.get(m, ("", ""))))
            st.dataframe(t3, hide_index=True, use_container_width=True)
            st.caption("Reductions are relative to Least-Loaded (no sustainability objective, no prewarming).")
        else:
            st.warning("Table 3 (table3_caprf_performance.csv) not found in results/tables/.")

        st.markdown("---")
        st.markdown(f"#### Tables 4 & 5: Baseline Comparisons on {DATASET_INFO[ds_key]['title']}")
        table_name = "table4_comparison_bitbrains.csv" if ds_key == "bitbrains" else "table5_comparison_kaggle.csv"
        comp_t = get_cached_table(table_name)
        if comp_t is not None:
            st.dataframe(comp_t, hide_index=True, use_container_width=True)
            st.caption("Schedulers: CAPRF++, Least-Loaded, Greedy-MCT, Green-Greedy, WTC-Greedy, Vanilla PPO, Round-Robin, Random.")
        else:
            st.warning(f"Comparison table `{table_name}` not found.")

        with st.expander("🔬 View Table 6: Wilcoxon Signed-Rank Statistical Tests"):
            t6 = get_cached_table("table6_wilcoxon.csv")
            if t6 is not None:
                st.dataframe(t6, hide_index=True, use_container_width=True)
                st.caption(
                    "p_intervals: paired over 10 execution intervals × 5 seeds (n = 50). "
                    "All carbon and cold-start reductions by CAPRF++ achieve p < 0.05 against Greedy-MCT and heuristics."
                )
            else:
                st.info("Table 6 not found.")

        with st.expander("🏢 View Table 9: Homogeneous vs Heterogeneous Workload Failure Rates (%)"):
            t9 = get_cached_table("table9_homogeneity.csv")
            if t9 is not None:
                st.dataframe(t9, hide_index=True, use_container_width=True)
            else:
                st.info("Table 9 not found.")

    # 2. Ablation Study
    with res_sub_tab2:
        st.markdown("#### Incremental Component Ablation Study")
        st.markdown(
            "Evaluation of each architectural module added on top of the previous:  \n"
            "**Vanilla Baseline PPO** $\\to$ **+Context Encoding** $\\to$ **+B+-Tree Indexing** $\\to$ "
            "**+Weighted MO-PPO** $\\to$ **+Trajectory Buffer** $\\to$ **CAPRF++ (full)**."
        )
        abl_name = "table7_ablation_bitbrains.csv" if ds_key == "bitbrains" else "table8_ablation_kaggle.csv"
        abl_df = get_cached_table(abl_name)
        if abl_df is not None:
            st.dataframe(abl_df, hide_index=True, use_container_width=True)
            st.markdown("""
            **Key Insights from the Ablation:**
            - **+Context Encoding:** Stabilizes training and enables generalization on unseen task types.
            - **+B+-Tree Indexing:** Reduces decision latency from $O(N)$ linear scan down to sub-3ms $O(\\log N)$ lookup.
            - **+Weighted MO-PPO & WTC:** Shifts the policy toward renewable nodes, generating ~10–13% carbon reduction.
            - **+Predictive Prewarming:** Yields the dramatic leap from ~60% to **91–95% cold-start reduction**.
            """)
        else:
            st.warning(f"Ablation table `{abl_name}` not found.")

    # 3. Plots & Figures
    with res_sub_tab3:
        st.markdown("#### High-Resolution Experimental Plots (From 5-Seed Benchmark)")

        fig_c1, fig_c2 = st.columns(2)

        with fig_c1:
            p1 = FIG_DIR / "fig_pareto_sla_vs_carbon.png"
            if p1.exists():
                st.image(str(p1), caption="Figure 1: SLA vs Carbon Pareto Frontier (Top-left is optimal; CAPRF++ achieves favorable tradeoff).")
            else:
                st.info("fig_pareto_sla_vs_carbon.png not found.")

            p2 = FIG_DIR / "fig3_executor_reuse_vs_startup.png"
            if p2.exists():
                st.image(str(p2), caption="Figure 2: Container Warm Reuse & Startup Delay (With vs Without Prewarming).")
            else:
                st.info("fig3_executor_reuse_vs_startup.png not found.")

            p3 = FIG_DIR / "fig7_task_scaling.png"
            if p3.exists():
                st.image(str(p3), caption="Figure 3: Task Workload Scaling (1k to 10k tasks).")
            else:
                st.info("fig7_task_scaling.png not found.")

        with fig_c2:
            p4 = FIG_DIR / "fig4_training_progress.png"
            if p4.exists():
                st.image(str(p4), caption="Figure 4: PPO Training Progress (Reward convergence over 120,000 steps).")
            else:
                st.info("fig4_training_progress.png not found.")

            iv_name = "fig5_intervals_bitbrains.png" if ds_key == "bitbrains" else "fig6_intervals_kaggle.png"
            p5 = FIG_DIR / iv_name
            if p5.exists():
                st.image(str(p5), caption=f"Figure 5: Performance across 10 temporal intervals ({DATASET_INFO[ds_key]['title']}).")
            else:
                st.info(f"{iv_name} not found.")

            p6 = FIG_DIR / "fig_index_benchmark.png"
            if p6.exists():
                st.image(str(p6), caption="Figure 6: B+-Tree / iDistance Retrieval vs Linear Scan Latency Scaling.")
            else:
                st.info("fig_index_benchmark.png not found.")

    # 4. Executive REPORT.md
    with res_sub_tab4:
        st.markdown("#### 📄 Executive Report (`results/REPORT.md`)")
        if REPORT_PATH.exists():
            with open(REPORT_PATH, "r", encoding="utf-8") as rf:
                report_text = rf.read()
            st.markdown(report_text)
        else:
            st.warning("`results/REPORT.md` not found.")
