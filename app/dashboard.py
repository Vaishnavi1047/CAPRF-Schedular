"""CAPRF++ presentation dashboard.

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from caprf.config import ABLATION
from caprf.data.workload import rescale_load
from caprf.experiments.common import RESULTS, dataset_config, load_splits, make_cluster
from caprf.experiments.run_all import load_caprf, run_dir
from caprf.schedulers.heuristics import HEURISTICS
from caprf.sim.runner import per_interval, run_episode, summarize

torch.set_num_threads(1)
st.set_page_config(page_title="CAPRF++ Scheduler", page_icon="⚙️", layout="wide")

DS = {"bitbrains": "Dataset 1 — GWA-T-12 Bitbrains (real trace)",
      "kaggle": "Dataset 2 — Kaggle Cloud Task Scheduling"}
COLORS = {"CAPRF++": "#d6336c", "Greedy-MCT": "#1c7ed6", "WTC-Greedy": "#f59f00", "Green-Greedy": "#37b24d",
          "Least-Loaded": "#7048e8", "Round-Robin": "#868e96", "Random": "#adb5bd"}
FIG, TAB = RESULTS / "figures", RESULTS / "tables"
HEUR = {H.name: H for H in HEURISTICS}


# ------------------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner="Loading trained CAPRF++ model…")
def model(ds: str, seed: int):
    return load_caprf(ds, seed)


@st.cache_resource(show_spinner="Loading workload…")
def splits(ds: str):
    return load_splits(dataset_config(ds))


def make_sched(name: str, ds: str, seed: int):
    return model(ds, seed)[0].eval_copy() if name == "CAPRF++" else HEUR[name]()


def available_seeds(ds: str) -> list[int]:
    base = run_dir(ds, ABLATION[-1].name, 0).parent
    return sorted(int(p.name[4:]) for p in base.glob("seed*") if (p / "model.pt").exists())


# ------------------------------------------------------------------------------ sidebar
st.sidebar.title("⚙️ CAPRF++")
st.sidebar.caption("Context-aware B+-tree scheduler with weighted multi-objective PPO and predictive prewarming")
ds = st.sidebar.selectbox("Dataset", list(DS), format_func=lambda d: DS[d])
seeds = available_seeds(ds)
if not seeds:
    st.error("No trained models found. Run `python -m caprf.experiments.run_all` first.")
    st.stop()
seed = st.sidebar.selectbox(
    "Trained model (seed)", seeds,
    help="CAPRF++ was trained 5 times with different random seeds. The seed changes the initial network "
         "weights, the training episodes drawn and their load, exploration, and the simulator's runtime noise. "
         "The tasks, held-out types, cluster hardware and hyper-parameters are identical. "
         "Results are reported as mean ± std over the 5 models; any of them is fine for a demo.")
st.sidebar.markdown("---")
st.sidebar.markdown("Paper: *Felsy & Isaac Sajan, Real-Time Systems 62:2 (2026)*  \n"
                    "doi:10.1007/s11241-026-09456-0")

tab_over, tab_res, tab_live, tab_insp = st.tabs(
    ["🧭 Overview", "📊 Results", "▶️ Live simulation", "🔍 Decision inspector"])

# ============================================================================== OVERVIEW
with tab_over:
    st.header("CAPRF++ — what it does")
    st.markdown(
        "For every job arriving at a cloud-edge system, CAPRF++ decides **which machine runs it**. "
        "It tries to meet the job's deadline (SLA) while minimising energy, carbon emissions and container "
        "cold-start delay, and to keep working on **task types it has never seen in training**.")
    st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
 a [label="Task arrives\\n(deadline, compute,\\npriority, sustainability, type)", fillcolor="#e7f5ff"];
 b [label="① Context-aware\\nfeature encoding\\nz-score + type embedding\\n→ z_t  (Eq. 1–6)", fillcolor="#d0ebff"];
 c [label="② B+-tree retrieval\\niDistance range query\\n‖z_t − z_r‖ ≤ τ\\n+ availability B+-tree  (Eq. 7–9)", fillcolor="#d3f9d8"];
 d [label="③ Weighted Tradeoff\\nController\\nΩ = αL+βE+δC+ρO\\nkeep Ω ≤ Ω_min+Δ  (Eq. 10–11)", fillcolor="#fff3bf"];
 e [label="④ PPO policy\\npicks machine R*\\n(Eq. 12–15)", fillcolor="#ffe8cc"];
 f [label="⑤ Predictive prewarming\\nλ·Cold(R) ≥ T_warm\\n(Eq. 16–18)", fillcolor="#ffe3e3"];
 g [label="Execute on\\nedge / cloud", fillcolor="#f3d9fa"];
 h [label="Reward (Eq. 13)\\nSLA, latency, energy,\\ncarbon, cold start\\n→ PPO update + FIFO buffer", fillcolor="#e5dbff"];
 a->b->c->d->e->f->g->h; h->e [style=dashed, label="learn"]; }
""")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Paper → code")
        st.dataframe(pd.DataFrame([
            ("Eq. 1 z-score (moving window)", "caprf/models/networks.py · RunningNorm"),
            ("Eq. 2–6 type embedding + 2-layer encoder", "caprf/models/networks.py · TaskEncoder"),
            ("Eq. 7 resource encoder ψ", "caprf/models/networks.py · ResourceEncoder"),
            ("Eq. 8–9 Euclidean match, adaptive τ, fallback", "caprf/index/bplustree.py · AdaptiveRetriever"),
            ("B+-tree index (O(log N))", "caprf/index/bplustree.py · BPlusTree, IDistanceIndex"),
            ("Eq. 10–11 WTC score Ω and feasible set", "caprf/schedulers/caprf.py · _tradeoff, _feasible"),
            ("Eq. 12 state, Eq. 13 reward", "caprf/schedulers/caprf.py · decide, observe"),
            ("Eq. 14–15 clipped PPO + GAE (Alg. 2)", "caprf/agent/ppo.py · PPOLearner, gae"),
            ("FIFO trajectory buffer + PER", "caprf/agent/ppo.py · Storage, update"),
            ("Eq. 16–18 arrival predictor + prewarming", "caprf/models/networks.py · ArrivalHead; caprf.py"),
            ("Cloud-edge simulator (Table 2)", "caprf/sim/cluster.py · Cluster"),
            ("Hyper-parameters (Table 1)", "caprf/config.py"),
        ], columns=["Paper", "Code"]), hide_index=True, use_container_width=True)
    with c2:
        st.subheader("Workload")
        sp = splits(ds)
        f = sp.full
        types = pd.DataFrame({"type": f.type_names, "tasks": np.bincount(f.ttype, minlength=len(f.type_names))})
        types["held out (unseen)"] = [i in f.unseen for i in range(len(types))]
        m1, m2, m3 = st.columns(3)
        m1.metric("Tasks", f"{len(f):,}")
        m2.metric("Train / test", f"{len(sp.train):,} / {len(sp.test):,}")
        m3.metric("Task types", f"{len(types)} ({len(f.unseen)} unseen)")
        st.dataframe(types, hide_index=True, use_container_width=True)
        st.caption("Task types are k-means workload classes. Unseen types are removed from training entirely "
                   "and only appear in the test window — that is how generalisation is measured.")
        st.markdown("**Cluster:** 150 edge + 50 cloud nodes, parameters from Table 2 of the paper "
                    "(cores, GHz, bandwidth, 50–200 ms cold start, 0.1–0.5 kg CO₂/kWh).")

# ============================================================================== RESULTS
with tab_res:
    st.header("Experimental results (5 seeds per configuration)")
    if not (TAB / "table3_caprf_performance.csv").exists():
        st.warning("Run `python -m caprf.experiments.analyze` to build the result tables.")
    else:
        st.subheader("Table 3 — CAPRF++ on both datasets")
        t3 = pd.read_csv(TAB / "table3_caprf_performance.csv")
        paper = {"Avg. task completion latency (ms)": ("25.0", "30.0"), "SLA satisfaction (%)": ("99.4", "99.3"),
                 "Carbon footprint reduction (%)": ("70.2", "70.8"),
                 "Cold-start overhead reduction (%)": ("54.1", "51.6"),
                 "Generalization on unseen workloads (%)": ("97.8", "97.1"), "Makespan (s)": ("340.3", "210.6")}
        t3["Paper (D1 / D2)"] = t3.Metric.map(lambda m: " / ".join(paper.get(m, ("", ""))))
        st.dataframe(t3, hide_index=True, use_container_width=True)
        st.caption("Ours are mean ± std over seeds in our simulator; reductions are vs. the Least-Loaded "
                   "balancer (no sustainability objective, no prewarming). The paper's numbers come from its own "
                   "unpublished simulator, so absolute values are not comparable.")

        st.subheader("Tables 4 & 5 — comparison with baseline schedulers")
        name = "table4_comparison_bitbrains.csv" if ds == "bitbrains" else "table5_comparison_kaggle.csv"
        st.dataframe(pd.read_csv(TAB / name), hide_index=True, use_container_width=True)
        if (FIG / "fig_pareto_sla_vs_carbon.png").exists():
            st.image(str(FIG / "fig_pareto_sla_vs_carbon.png"),
                     caption="SLA vs carbon: top-left is best. Greedy schedulers sit at one extreme each; "
                             "CAPRF++ trades a little SLA for lower carbon and far fewer cold starts.")

        st.subheader("Tables 7 & 8 — ablation (each module added on top of the previous)")
        name = "table7_ablation_bitbrains.csv" if ds == "bitbrains" else "table8_ablation_kaggle.csv"
        st.dataframe(pd.read_csv(TAB / name), hide_index=True, use_container_width=True)
        st.caption("'Baseline' is vanilla PPO with raw features, random candidates and an SLA+latency reward. "
                   "The multi-objective step moves the policy toward lower carbon; prewarming adds the "
                   "large cold-start reduction.")

        with st.expander("Table 6 — Wilcoxon signed-rank tests"):
            st.dataframe(pd.read_csv(TAB / "table6_wilcoxon.csv"), hide_index=True, use_container_width=True)
            st.caption("p_intervals: paired over 10 execution intervals × 5 seeds (n = 50). "
                       "p_seeds_two_sided: paired over 5 seeds — its exact minimum is 2/2⁵ = 0.0625, "
                       "so with 5 runs no two-sided test can reach p < 0.05.")
        if (TAB / "table9_homogeneity.csv").exists():
            with st.expander("Table 9 — failure rate on homogeneous vs heterogeneous workloads"):
                st.dataframe(pd.read_csv(TAB / "table9_homogeneity.csv"), hide_index=True, use_container_width=True)

        st.subheader("Figures")
        figs = [
            ("fig4_training_progress.png", "Training progress: reward per task rises as PPO learns. Average task time does "
             "not fall — the reward also prices energy, carbon and cold starts, so the agent learns to accept some "
             "latency for them; every episode also runs at a random load, which makes the curves noisy."),
            ("fig3_executor_reuse_vs_startup.png", "Executor (warm container) reuse and startup delay, with vs without "
             "prewarming. The Bitbrains dip near task 13.5k is a traffic burst in the real trace: tasks spill onto "
             "machines without a warm container."),
            (f"fig{5 if ds == 'bitbrains' else 6}_intervals_{ds}.png", "Per execution interval: latency, SLA and carbon for each scheduler."),
            ("fig7_task_scaling.png", "Scaling with the number of tasks (1k–10k) at the same load."),
            ("fig8_device_scaling.png", "Scaling with cluster size (100–1000 devices), workload scaled proportionally."),
            ("fig_index_benchmark.png", "B+-tree/iDistance retrieval vs exhaustive scan as the resource pool grows."),
        ]
        for fn, cap in figs:
            if (FIG / fn).exists():
                st.image(str(FIG / fn), caption=cap)

# ============================================================================== LIVE SIMULATION
with tab_live:
    st.header("Run the trained scheduler live")
    sp = splits(ds)
    cfg = dataset_config(ds)
    c1, c2, c3 = st.columns([1, 1, 2])
    n = c1.slider("Tasks", 500, len(sp.test), min(4000, len(sp.test)), step=500)
    load = c2.slider("Cluster load", 0.4, 1.0, float(cfg.workload.target_load), step=0.05,
                     help="Mean utilisation the arrival rate is scaled to. Higher = more queueing.")
    chosen = c3.multiselect("Schedulers", ["CAPRF++"] + list(HEUR),
                            default=["CAPRF++", "Greedy-MCT", "Least-Loaded", "WTC-Greedy", "Round-Robin"])
    if st.button("▶ Run simulation", type="primary") and chosen:
        w = rescale_load(sp.test.window(0, n), cfg.cluster, load)
        results, bar = {}, st.progress(0.0, text="starting…")
        for k, name in enumerate(chosen):
            def prog(i, total, k=k, name=name):
                bar.progress((k + i / total) / len(chosen), text=f"{name}: task {i:,}/{total:,}")
            results[name] = run_episode(make_sched(name, ds, seed), make_cluster(cfg, noise_seed=seed), w,
                                        progress=prog)
        bar.progress(1.0, text="done")
        st.session_state.live = (results, n, load)

    if "live" in st.session_state:
        results, n_run, load_run = st.session_state.live
        st.caption(f"{n_run:,} test tasks at load {load_run:.2f}")
        summ = {k: summarize(v) for k, v in results.items()}
        cols = st.columns(len(summ))
        for col, (name, s) in zip(cols, summ.items()):
            col.markdown(f"**<span style='color:{COLORS.get(name)}'>{name}</span>**", unsafe_allow_html=True)
            col.metric("SLA met", f"{s.sla:.1f}%")
            col.metric("Avg latency", f"{s.latency_ms:,.0f} ms")
            col.metric("Carbon", f"{s.carbon_g:.1f} g")
            col.metric("Energy", f"{s.energy_kj:,.0f} kJ")
            col.metric("Cold-start time", f"{s.cold_start_s:.1f} s")
        table = pd.DataFrame({k: v.as_dict() for k, v in summ.items()}).T.round(2)
        with st.expander("All metrics"):
            st.dataframe(table, use_container_width=True)

        iv = pd.concat([per_interval(df).assign(scheduler=k) for k, df in results.items()]).reset_index()
        iv["interval"] += 1
        g1, g2, g3 = st.columns(3)
        for col, metric, title, log in [(g1, "latency_ms", "Avg latency (ms)", True), (g2, "sla", "SLA met (%)", False),
                                        (g3, "carbon_g", "Carbon (g CO₂)", False)]:
            fig = px.line(iv, x="interval", y=metric, color="scheduler", markers=True, log_y=log,
                          color_discrete_map=COLORS, title=title)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), legend_title=None)
            col.plotly_chart(fig, use_container_width=True)

        p1, p2 = st.columns(2)
        sc = pd.DataFrame([dict(scheduler=k, sla=s.sla, carbon=s.carbon_g, cold=s.cold_start_s) for k, s in summ.items()])
        fig = px.scatter(sc, x="carbon", y="sla", color="scheduler", size="cold", size_max=40,
                         color_discrete_map=COLORS, title="Trade-off: SLA vs carbon (bubble = cold-start time)")
        fig.update_layout(height=360, xaxis_title="carbon (g) ← better", yaxis_title="SLA % ↑ better")
        p1.plotly_chart(fig, use_container_width=True)
        split = pd.DataFrame([dict(scheduler=k, edge=100 - s.cloud_share, cloud=s.cloud_share) for k, s in summ.items()])
        fig = go.Figure([go.Bar(name="edge", x=split.scheduler, y=split.edge, marker_color="#37b24d"),
                         go.Bar(name="cloud", x=split.scheduler, y=split.cloud, marker_color="#1c7ed6")])
        fig.update_layout(barmode="stack", height=360, title="Where tasks ran (% of tasks)")
        p2.plotly_chart(fig, use_container_width=True)

# ============================================================================== INSPECTOR
with tab_insp:
    st.header("Step through CAPRF++ decisions")
    st.caption("Each step schedules the next test task and shows every stage of the pipeline for it.")
    sp = splits(ds)
    cfg = dataset_config(ds)
    key = (ds, seed)
    if st.session_state.get("insp_key") != key or st.button("↺ Reset"):
        w = rescale_load(sp.test, cfg.cluster, cfg.workload.target_load)
        sched = make_sched("CAPRF++", ds, seed)
        cl = make_cluster(cfg, noise_seed=seed)
        cl.reset()
        sched.begin(cl, w, train=False)
        st.session_state.update(insp_key=key, insp=dict(w=w, sched=sched, cl=cl, i=0, last=None))
    S = st.session_state.insp

    def advance(steps: int, until_unseen: bool = False):
        w, sched, cl = S["w"], S["sched"], S["cl"]
        done = 0
        while S["i"] < len(w):
            i = S["i"]
            cl.now = float(w.arrival[i])
            j = sched.decide(cl, w, i)
            trace = dict(sched.debug)
            rec = cl.assign(w, i, j)
            sched.observe(cl, w, i, rec)
            S["i"] += 1
            done += 1
            S["last"] = (i, trace, rec)
            if (until_unseen and w.is_unseen()[i]) or (not until_unseen and done >= steps):
                break

    b = st.columns(5)
    if b[0].button("Next task ▶", type="primary"):
        advance(1)
    if b[1].button("Skip 10"):
        advance(10)
    if b[2].button("Skip 500"):
        advance(500)
    if b[3].button("Next unseen-type task"):
        advance(1, until_unseen=True)
    b[4].markdown(f"**Scheduled:** {S['i']:,} / {len(S['w']):,}")

    if S["last"] is None:
        st.info("Press **Next task** to schedule the first task.")
    else:
        i, tr, rec = S["last"]
        w, cl = S["w"], S["cl"]
        tname = w.type_names[int(w.ttype[i])]
        st.subheader(f"Task #{i}")
        c = st.columns(7)
        c[0].metric("Type", tname.split(":")[0], "UNSEEN" if w.is_unseen()[i] else "seen in training",
                    delta_color="inverse" if w.is_unseen()[i] else "off")
        c[1].metric("Deadline", f"{w.deadline[i]*1000:,.0f} ms")
        c[2].metric("Compute", f"{w.cycles[i]:.2f} Gcycles")
        c[3].metric("Cores", int(w.cores[i]))
        c[4].metric("Data", f"{w.data_mb[i]:.2f} MB")
        c[5].metric("Priority", {1: "high", 2: "medium", 3: "low"}[int(w.priority[i])])
        c[6].metric("Sustainability pref.", f"{w.sustain[i]:.2f}")

        cand, pred, omega, sel = tr["cand"], tr["pred"], tr["omega"], tr["sel"]
        st.markdown(f"**① + ② Retrieval:** the B+-tree index returned **{len(cand)} of {cl.n}** machines "
                    f"(τ = {(tr['tau'] or 0):.3f}). **③ WTC** kept **{len(sel)}** within Ω_min + Δ "
                    f"(and meeting the deadline when possible).")
        df = pd.DataFrame(dict(node=cand, kind=np.where(cl.is_cloud[cand], "cloud", "edge"),
                               warm=pred["warm"], latency_ms=1000 * pred["latency"], energy_J=pred["energy"],
                               carbon_mg=1000 * pred["carbon"], cold_ms=1000 * pred["cold"], omega=omega))
        df["meets_deadline"] = pred["latency"] <= w.deadline[i]
        df["feasible (C_i)"] = False
        df.loc[sel, "feasible (C_i)"] = True
        df["policy_prob"] = np.nan
        df.loc[sel, "policy_prob"] = tr["probs"]
        df["decision"] = ""
        df.loc[df.node == tr["chosen"], "decision"] = "✅ chosen"
        df.loc[df.node.isin(tr["prewarmed"]), "decision"] = "🔥 prewarmed"
        df = df.sort_values(["feasible (C_i)", "policy_prob", "omega"], ascending=[False, False, True])
        st.dataframe(df.round(3), hide_index=True, use_container_width=True, height=320)

        l, r = st.columns([3, 2])
        feas = df[df["feasible (C_i)"]].copy()
        feas["label"] = feas.node.astype(str) + " (" + feas.kind + ")"
        fig = px.bar(feas, x="label", y="policy_prob", color="omega", color_continuous_scale="RdYlGn_r",
                     title="④ PPO policy over the feasible set (colour = Ω, lower is better)")
        fig.update_layout(height=320, xaxis_title="machine", yaxis_title="probability")
        l.plotly_chart(fig, use_container_width=True)
        with r:
            st.markdown("**⑤ Prewarming**")
            if tr["lam"] is None:
                st.write("Not evaluated (single feasible machine).")
            else:
                st.write(f"Predicted arrival probability of another `{tname.split(':')[0]}` task soon: "
                         f"**λ = {tr['lam']:.2f}**. Prewarm a top-ranked machine if λ·Cold(R) ≥ "
                         f"{cfg.caprf.prewarm_threshold}.")
                st.write("Prewarmed: " + (", ".join(f"node {n_}" for n_ in tr["prewarmed"]) or "none"))
            st.markdown("**Outcome**")
            st.write(f"Ran on **node {rec.node}** ({'cloud' if cl.is_cloud[rec.node] else 'edge'}): "
                     f"latency **{rec.latency*1000:,.0f} ms** "
                     f"(queue {rec.wait*1000:,.0f}, cold start {rec.cold_s*1000:,.0f}, exec {rec.exec_s*1000:,.0f}) — "
                     + ("✅ deadline met" if rec.sla else "❌ deadline missed")
                     + f"; energy {rec.energy_j:.1f} J, carbon {rec.carbon_g*1000:.2f} mg; "
                     + ("warm container reused" if rec.warm_hit else "cold start"))
