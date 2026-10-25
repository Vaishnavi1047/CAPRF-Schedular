"""Build the paper-style tables, figures, statistics and REPORT.md from results/runs/.

    python -m caprf.experiments.analyze
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from caprf.config import ABLATION
from caprf.experiments.common import RESULTS
from caprf.experiments.run_all import DATASETS, RUNS, run_dir, slug
from caprf.schedulers.heuristics import HEURISTICS
from caprf.sim.runner import per_interval

FIG, TAB, EXP = RESULTS / "figures", RESULTS / "tables", RESULTS / "experiments"
OURS = "CAPRF++"
FULL_NAME = ABLATION[-1].name
REFERENCE = "Least-Loaded"  # classic load balancing: no sustainability objective, no prewarming (Sec. 4.3.1)
DS_LABEL = {"bitbrains": "Dataset 1 (GWA-T-12 Bitbrains)", "kaggle": "Dataset 2 (Kaggle Cloud Task Scheduling)"}
ORDER = [OURS] + [H.name for H in HEURISTICS] + ["Vanilla PPO"]
COLORS = {OURS: "#d6336c", "Greedy-MCT": "#1c7ed6", "WTC-Greedy": "#f59f00", "Green-Greedy": "#37b24d",
          "Least-Loaded": "#7048e8", "Round-Robin": "#868e96", "Random": "#adb5bd", "Vanilla PPO": "#0ca678"}
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def _name_for(dirname: str) -> str:
    for v in ABLATION:
        if slug(v.name) == dirname:
            return OURS if v.name == FULL_NAME else ("Vanilla PPO" if v.name == "Baseline" else v.name)
    for H in HEURISTICS:
        if slug(H.name) == dirname:
            return H.name
    return dirname


def load_summaries() -> pd.DataFrame:
    rows = []
    for f in RUNS.glob("*/*/seed*/summary.json"):
        d = json.loads(f.read_text())
        d.update(dataset=f.parts[-4], scheduler=_name_for(f.parts[-3]), dir=f.parts[-3],
                 seed=int(f.parts[-2][4:]))
        rows.append(d)
    df = pd.DataFrame(rows)
    ref = df[df.scheduler == REFERENCE].set_index(["dataset", "seed"])
    key = list(zip(df.dataset, df.seed))
    df["carbon_reduction"] = 100 * (1 - df.carbon_g.to_numpy() / ref.loc[key, "carbon_g"].to_numpy())
    df["cold_start_reduction"] = 100 * (1 - df.cold_start_s.to_numpy() / ref.loc[key, "cold_start_s"].to_numpy())
    df["energy_reduction"] = 100 * (1 - df.energy_kj.to_numpy() / ref.loc[key, "energy_kj"].to_numpy())
    return df


def pm(s: pd.Series, d=1) -> str:
    return f"{s.mean():.{d}f} ± {s.std(ddof=1):.{d}f}" if len(s) > 1 else f"{s.mean():.{d}f}"


METRICS = [("latency_ms", "Avg. task completion latency (ms)", 0), ("sla", "SLA satisfaction (%)", 1),
           ("carbon_reduction", "Carbon footprint reduction (%)", 1),
           ("cold_start_reduction", "Cold-start overhead reduction (%)", 1),
           ("generalization", "Generalization on unseen workloads (%)", 1), ("makespan_s", "Makespan (s)", 1)]


def table_main(df):
    out = {}
    for ds in DATASETS:
        d = df[(df.dataset == ds) & (df.scheduler == OURS)]
        out[DS_LABEL[ds]] = {lab: pm(d[m], p) for m, lab, p in METRICS}
    t = pd.DataFrame(out)
    t.index.name = "Metric"
    return t


def table_compare(df, ds):
    d = df[(df.dataset == ds) & df.scheduler.isin(ORDER)]
    cols = [("latency_ms", "Latency (ms)", 0), ("sla", "SLA (%)", 1), ("carbon_g", "Carbon (g CO2)", 1),
            ("carbon_reduction", "Carbon red. (%)", 1), ("energy_kj", "Energy (kJ)", 0),
            ("cold_start_reduction", "Cold-start red. (%)", 1), ("warm_hit", "Executor reuse (%)", 1),
            ("generalization", "Generalization (%)", 1), ("makespan_s", "Makespan (s)", 1)]
    rows = {}
    for s in [x for x in ORDER if x in set(d.scheduler)]:
        g = d[d.scheduler == s]
        rows[s] = {lab: pm(g[m], p) for m, lab, p in cols}
    t = pd.DataFrame(rows).T
    t.index.name = "Scheduler"
    return t


def table_ablation(df, ds):
    rows = {}
    for v in ABLATION:
        g = df[(df.dataset == ds) & (df.dir == slug(v.name))]
        if len(g):
            rows[v.name] = {lab: pm(g[m], p) for m, lab, p in METRICS}
    t = pd.DataFrame(rows).T
    t.index.name = "Model configuration"
    return t


def _intervals(ds, name, seed):
    f = run_dir(ds, name, seed) / "tasks.csv.gz"
    return per_interval(pd.read_csv(f)) if f.exists() else None


def table_stats(df):
    """Wilcoxon signed-rank tests, CAPRF++ vs the strongest baselines.

    (a) paired over seeds (n = #seeds). With n = 5 the exact two-sided test cannot go below
        p = 2/2^5 = 0.0625 no matter how large the effect, so (b) is the informative one:
    (b) paired over 10 execution intervals x seeds (n = 50), same workload slice and noise seed.
    """
    rows = []
    metrics = [("latency_ms", "less"), ("sla", "greater"), ("carbon_g", "less"), ("cold_ms", "less")]
    for ds in DATASETS:
        seeds = sorted(df[(df.dataset == ds) & (df.scheduler == OURS)].seed)
        for base in ["Greedy-MCT", "WTC-Greedy", "Green-Greedy"]:
            for m, alt in metrics:
                a_iv, b_iv, a_sd, b_sd = [], [], [], []
                for s in seeds:
                    ia, ib = _intervals(ds, FULL_NAME, s), _intervals(ds, base, s)
                    if ia is None or ib is None:
                        continue
                    a_iv += ia[m].tolist(); b_iv += ib[m].tolist()
                    a_sd.append(ia[m].mean() if m != "carbon_g" else ia[m].sum())
                    b_sd.append(ib[m].mean() if m != "carbon_g" else ib[m].sum())
                if len(a_iv) < 6:
                    continue
                p_iv = wilcoxon(a_iv, b_iv, alternative=alt).pvalue
                p_sd = wilcoxon(a_sd, b_sd, alternative="two-sided").pvalue if len(a_sd) >= 2 else np.nan
                rows.append(dict(dataset=ds, baseline=base, metric=m, hypothesis=f"CAPRF++ {alt} than baseline",
                                 caprf_mean=np.mean(a_iv), baseline_mean=np.mean(b_iv),
                                 n_intervals=len(a_iv), p_intervals=p_iv,
                                 n_seeds=len(a_sd), p_seeds_two_sided=p_sd,
                                 significant_0_05=p_iv < 0.05))
    return pd.DataFrame(rows)


def table_homogeneity():
    fs = sorted(EXP.glob("homogeneity_*_seed*.csv"))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f).assign(dataset=f.stem.split("_")[1]) for f in fs])
    d["scheduler"] = d.scheduler.replace({OURS: OURS})
    t = d.groupby(["dataset", "scheduler", "workload"]).failure_rate.mean().unstack("workload").round(2)
    return t[["homogeneous", "heterogeneous"]]


# ---------------------------------------------------------------------------- figures
def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, ds in zip(axes, DATASETS):
        logs = [pd.read_csv(f) for f in sorted((RUNS / ds / slug(FULL_NAME)).glob("seed*/train_log.csv"))]
        if not logs:
            continue
        n = min(len(l) for l in logs)
        lat = np.stack([l.latency_ms[:n] for l in logs]); rew = np.stack([l.reward[:n] / 4000 for l in logs])
        x = logs[0].steps[:n] / 1000
        ax.plot(x, lat.mean(0), color="#1c7ed6", marker="o", ms=3, label="avg task time (ms)")
        ax.fill_between(x, lat.mean(0) - lat.std(0), lat.mean(0) + lat.std(0), color="#1c7ed6", alpha=0.15)
        ax.set_xlabel("training steps (×1000)"); ax.set_ylabel("avg task completion time (ms)")
        ax2 = ax.twinx()
        ax2.plot(x, rew.mean(0), color="#f76707", marker="s", ms=3, label="reward / task")
        ax2.fill_between(x, rew.mean(0) - rew.std(0), rew.mean(0) + rew.std(0), color="#f76707", alpha=0.15)
        ax2.set_ylabel("mean reward per task"); ax2.grid(False)
        ax.set_title(f"{DS_LABEL[ds]} — mean ± std over {len(logs)} seeds")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    _save(fig, "fig4_training_progress.png")


def fig_reuse():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, ds in zip(axes, DATASETS):
        for name, ls, lab in [(FULL_NAME, "-", "with prewarming"), ("+Trajectory Buffer", "--", "without prewarming")]:
            f = run_dir(ds, name, 0) / "tasks.csv.gz"
            if not f.exists():
                continue
            t = pd.read_csv(f).sort_values("start")
            g = np.arange(len(t)) * 10 // len(t)
            agg = t.groupby(g).agg(reuse=("warm_hit", "mean"), startup=("cold_s", lambda x: 1000 * x.mean()))
            x = (agg.index + 1) * len(t) / 10 / 1000
            ax.plot(x, agg.reuse, ls, color="#1c7ed6", marker="o", ms=3, label=f"executor reuse — {lab}")
            if ls == "-":
                ax2 = ax.twinx(); ax2.grid(False); ax2.set_ylabel("mean startup delay per task (ms)")
            ax2.plot(x, agg.startup, ls, color="#f76707", marker="s", ms=3, label=f"startup delay — {lab}")
        ax.set_xlabel("completed task index (×1000)"); ax.set_ylabel("executor reuse rate")
        ax.set_title(DS_LABEL[ds])
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    _save(fig, "fig3_executor_reuse_vs_startup.png")


def fig_intervals():
    for ds, num in zip(DATASETS, (5, 6)):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
        for name in ORDER:
            dirname = FULL_NAME if name == OURS else ("Baseline" if name == "Vanilla PPO" else name)
            ivs = [iv for s in range(5) if (iv := _intervals(ds, dirname, s)) is not None]
            if not ivs:
                continue
            m = pd.concat(ivs).groupby(level=0).mean()
            x = m.index + 1
            kw = dict(color=COLORS.get(name), marker="o", ms=3, lw=2.2 if name == OURS else 1.2, label=name)
            axes[0].plot(x, m.latency_ms, **kw); axes[1].plot(x, m.sla, **kw); axes[2].plot(x, m.carbon_g, **kw)
        for ax, t in zip(axes, ["(a) avg. task completion latency (ms)", "(b) SLA satisfaction (%)", "(c) carbon (g CO2)"]):
            ax.set_title(t); ax.set_xlabel("execution interval")
        axes[0].set_yscale("log")
        axes[1].legend(fontsize=7, ncol=2)
        fig.suptitle(DS_LABEL[ds], y=1.02)
        _save(fig, f"fig{num}_intervals_{ds}.png")


def fig_task_scaling():
    fs = sorted(EXP.glob("task_scaling_*_seed0.csv"))
    if not fs:
        return
    fig, axes = plt.subplots(len(fs), 3, figsize=(12, 3.2 * len(fs)), squeeze=False)
    for row, f in zip(axes, fs):
        d = pd.read_csv(f); ds = f.stem.split("_")[2]
        for name, g in d.groupby("scheduler"):
            if name in ("Random", "Round-Robin"):
                continue
            kw = dict(color=COLORS.get(name), marker="o", ms=3, lw=2.2 if name == OURS else 1.2, label=name)
            row[0].plot(g.n_tasks / 1000, g.latency_ms, **kw)
            row[1].plot(g.n_tasks / 1000, g.makespan_s, **kw)
            row[2].plot(g.n_tasks / 1000, g.sla, **kw)
        for ax, t in zip(row, ["(a) avg. latency (ms)", "(b) makespan (s)", "(c) SLA (%)"]):
            ax.set_title(f"{t} — {ds}"); ax.set_xlabel("completed task count (×1000)")
        row[2].legend(fontsize=7)
    _save(fig, "fig7_task_scaling.png")


def fig_device_scaling():
    fs = sorted(EXP.glob("device_scaling_*_seed0.csv"))
    if not fs:
        return
    d = pd.read_csv(fs[0])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for name, g in d.groupby("scheduler"):
        if name in ("Random", "Round-Robin"):
            continue
        kw = dict(color=COLORS.get(name), marker="o", ms=3, lw=2.2 if name == OURS else 1.2, label=name)
        axes[0].plot(g.devices, g.latency_ms, **kw); axes[1].plot(g.devices, g.utilization, **kw)
        axes[2].plot(g.devices, g.energy_kj, **kw)
    for ax, t in zip(axes, ["(a) avg. latency (ms)", "(b) device utilisation (%)", "(c) energy (kJ)"]):
        ax.set_title(t); ax.set_xlabel("number of devices")
    axes[0].legend(fontsize=7)
    _save(fig, "fig8_device_scaling.png")


def fig_index():
    f = EXP / "index_benchmark.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    axes[0].loglog(d.N, d.btree_query_ms, "o-", color=COLORS[OURS], label="B+-tree / iDistance query")
    axes[0].loglog(d.N, d.linear_python_ms, "s--", color="#868e96", label="linear scan (pure Python)")
    axes[0].loglog(d.N, d.linear_numpy_ms, "^:", color="#1c7ed6", label="linear scan (NumPy, vectorised)")
    axes[0].loglog(d.N, d.btree_update_ms, "d-", color="#37b24d", label="B+-tree key update")
    axes[0].set_xlabel("number of resources N"); axes[0].set_ylabel("ms per operation"); axes[0].legend(fontsize=7)
    axes[0].set_title("retrieval cost")
    axes[1].semilogx(d.N, 100 * d.frac_evaluated, "o-", color=COLORS[OURS])
    axes[1].set_xlabel("number of resources N"); axes[1].set_ylabel("% of resources whose distance is computed")
    axes[1].set_title("search-space pruning by the index")
    _save(fig, "fig_index_benchmark.png")


def fig_pareto(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, ds in zip(axes, DATASETS):
        d = df[(df.dataset == ds) & df.scheduler.isin(ORDER)]
        for name, g in d.groupby("scheduler"):
            if name in ("Random", "Round-Robin"):
                continue
            ax.errorbar(g.carbon_g.mean(), g.sla.mean(), xerr=g.carbon_g.std(), yerr=g.sla.std(), fmt="o",
                        color=COLORS.get(name), ms=8 if name == OURS else 6, capsize=3, label=name)
        ax.set_xlabel("carbon (g CO2)  ← better"); ax.set_ylabel("SLA satisfaction (%)  ↑ better")
        ax.set_title(f"SLA vs carbon trade-off — {ds}")
        ax.legend(fontsize=7)
    _save(fig, "fig_pareto_sla_vs_carbon.png")


# ---------------------------------------------------------------------------- report
def md(t: pd.DataFrame | None) -> str:
    return "_not run_" if t is None or len(t) == 0 else t.to_markdown()


def main():
    df = load_summaries()
    TAB.mkdir(parents=True, exist_ok=True)
    t3 = table_main(df); t3.to_csv(TAB / "table3_caprf_performance.csv")
    comp = {ds: table_compare(df, ds) for ds in DATASETS}
    for ds, t in comp.items():
        t.to_csv(TAB / f"table{4 if ds == 'bitbrains' else 5}_comparison_{ds}.csv")
    stats = table_stats(df); stats.to_csv(TAB / "table6_wilcoxon.csv", index=False)
    abl = {ds: table_ablation(df, ds) for ds in DATASETS}
    for ds, t in abl.items():
        t.to_csv(TAB / f"table{7 if ds == 'bitbrains' else 8}_ablation_{ds}.csv")
    homo = table_homogeneity()
    if homo is not None:
        homo.to_csv(TAB / "table9_homogeneity.csv")
    idx = pd.read_csv(EXP / "index_benchmark.csv") if (EXP / "index_benchmark.csv").exists() else None
    df.to_csv(TAB / "all_runs.csv", index=False)

    fig_training(); fig_reuse(); fig_intervals(); fig_task_scaling(); fig_device_scaling(); fig_index(); fig_pareto(df)

    dec = df[df.scheduler == OURS].groupby("dataset")[["decision_ms", "decision_ms_p99"]].mean().round(2)
    st = stats.copy()
    if len(st):
        st["p_intervals"] = st.p_intervals.map(lambda p: f"{p:.2e}")
        st["p_seeds_two_sided"] = st.p_seeds_two_sided.map(lambda p: f"{p:.4f}")
        st = st.round(3)
    report = f"""# CAPRF++ — experimental results

Generated by `python -m caprf.experiments.analyze` from {df.seed.nunique()} seed(s) per configuration.
All numbers come from our simulator (Sec. 4.1 of the paper, parameters of Tables 1-2) driven by the two
datasets; they are **not** expected to equal the paper's numbers. Reductions are relative to
{REFERENCE} (no sustainability objective, no prewarming), as in Sec. 4.3.1.

**Seeds.** A seed changes the randomness only: initial network weights, the training episodes drawn and
their load, exploration, UNK/replay sampling, iDistance reference points, and the simulator's runtime noise.
The workload, held-out task types, cluster hardware and hyper-parameters are identical across seeds.
Baselines are deterministic rules, so only runtime noise varies for them, hence their tiny ± values.

## Table 3 — CAPRF++ on both datasets (mean ± std over seeds)
{md(t3)}

## Table 4 — comparison on {DS_LABEL['bitbrains']}
{md(comp['bitbrains'])}

## Table 5 — comparison on {DS_LABEL['kaggle']}
{md(comp['kaggle'])}

![pareto](figures/fig_pareto_sla_vs_carbon.png)

## Table 6 — Wilcoxon signed-rank tests
`p_intervals`: paired over 10 execution intervals × seeds (one-sided, direction in *hypothesis*).
`p_seeds_two_sided`: paired over seeds only; with 5 pairs the exact two-sided minimum is 0.0625.

{md(st)}

## Tables 7/8 — ablation (modules added one at a time)
### {DS_LABEL['bitbrains']}
{md(abl['bitbrains'])}

### {DS_LABEL['kaggle']}
{md(abl['kaggle'])}

## Table 9 — scheduling failure rate (%) on homogeneous vs heterogeneous workloads
{md(homo)}

## Retrieval index benchmark (B+-tree / iDistance vs exhaustive scan)
{md(idx.round(4) if idx is not None else None)}

## Scheduling decision time of CAPRF++ (ms per task, full pipeline)
{md(dec)}

## Figures
![training](figures/fig4_training_progress.png)
![reuse](figures/fig3_executor_reuse_vs_startup.png)
![intervals1](figures/fig5_intervals_bitbrains.png)
![intervals2](figures/fig6_intervals_kaggle.png)
![tasks](figures/fig7_task_scaling.png)
![devices](figures/fig8_device_scaling.png)
![index](figures/fig_index_benchmark.png)
"""
    (RESULTS / "REPORT.md").write_text(report)
    print(report[:3000])


if __name__ == "__main__":
    main()
