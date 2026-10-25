"""Run every experiment of Sec. 4 and store raw results under results/runs/.

    python -m caprf.experiments.run_all                 # 5 seeds, both datasets, all experiments
    python -m caprf.experiments.run_all --seeds 1 --steps 40000   # quick pass

Afterwards build tables/figures/report with:  python -m caprf.experiments.analyze
"""
from __future__ import annotations

import os

# One maths-library thread per process: parallelism comes from running many worker processes,
# and letting each one also spawn a thread per core oversubscribes the CPU (~20x slowdown).
# Must be set before numpy/torch are imported; spawned workers inherit the environment.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from caprf.config import ABLATION, ClusterConfig
from caprf.data.workload import rescale_load
from caprf.experiments.common import (RESULTS, dataset_config, evaluate, load_splits, make_cluster,
                                      save_json, train_caprf)
from caprf.schedulers.caprf import CAPRFScheduler
from caprf.schedulers.heuristics import HEURISTICS
from caprf.sim.cluster import Cluster
from caprf.sim.runner import run_episode, summarize

RUNS = RESULTS / "runs"
DATASETS = ["bitbrains", "kaggle"]
FULL = len(ABLATION) - 1
TASK_COLS = ["task", "node", "arrival", "start", "finish", "latency", "wait", "exec_s", "cold_s",
             "energy_j", "carbon_g", "sla", "warm_hit", "ttype", "unseen", "is_cloud"]


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def run_dir(ds: str, name: str, seed: int) -> Path:
    return RUNS / ds / slug(name) / f"seed{seed}"


def _store(df: pd.DataFrame, out: Path, extra: dict | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    s = summarize(df).as_dict()
    s.update(extra or {})
    save_json(s, out / "summary.json")
    df[TASK_COLS].to_csv(out / "tasks.csv.gz", index=False, compression="gzip")
    return s


def load_caprf(ds: str, seed: int, variant_idx: int = FULL) -> tuple[CAPRFScheduler, object]:
    cfg = dataset_config(ds, seed, ABLATION[variant_idx])
    sp = load_splits(cfg)
    seen = sorted(set(np.unique(sp.train.ttype).tolist()))
    sched = CAPRFScheduler(cfg, len(sp.full.type_names), seen)
    sched.load(run_dir(ds, ABLATION[variant_idx].name, seed) / "model.pt")
    return sched, cfg


# ------------------------------------------------------------------------------- jobs
def job_train(ds: str, vi: int, seed: int, steps: int) -> str:
    torch.set_num_threads(1)
    v = ABLATION[vi]
    out = run_dir(ds, v.name, seed)
    if (out / "summary.json").exists():
        return f"skip {out}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = dataset_config(ds, seed, v)
    sp = load_splits(cfg)
    t0 = time.time()
    sched, log = train_caprf(cfg, sp, steps, verbose=False, log_path=out / "train_log.csv")
    sched.save(out / "model.pt")
    ev = sched.eval_copy()
    df = run_episode(ev, make_cluster(cfg, noise_seed=seed), sp.test)
    _store(df, out, dict(train_s=time.time() - t0, best_episode=getattr(sched, "best_episode", -1),
                         decision_ms=1000 * float(np.mean(ev.decision_times)),
                         decision_ms_p99=1000 * float(np.quantile(ev.decision_times, 0.99)),
                         retrieval_fallbacks=ev.retriever.fallbacks))
    return f"trained {ds}/{v.name}/seed{seed} in {time.time() - t0:.0f}s"


def job_heuristic(ds: str, hi: int, seed: int) -> str:
    H = HEURISTICS[hi]
    out = run_dir(ds, H.name, seed)
    if (out / "summary.json").exists():
        return f"skip {out}"
    cfg = dataset_config(ds, seed)
    sp = load_splits(cfg)
    df = run_episode(H(), make_cluster(cfg, noise_seed=seed), sp.test)
    _store(df, out)
    return f"heuristic {ds}/{H.name}/seed{seed}"


def _schedulers(ds: str, seed: int):
    sched, cfg = load_caprf(ds, seed)
    yield "CAPRF++", sched.eval_copy, cfg
    for H in HEURISTICS:
        yield H.name, H, cfg


def job_task_scaling(ds: str, seed: int) -> str:
    """Fig. 7: 1k..10k tasks at the target load."""
    torch.set_num_threads(1)
    out = RESULTS / "experiments" / f"task_scaling_{ds}_seed{seed}.csv"
    if out.exists():
        return f"skip {out}"
    rows = []
    for name, factory, cfg in _schedulers(ds, seed):
        sp = load_splits(cfg)
        for n in range(1000, 10001, 1000):
            w = rescale_load(sp.test.window(0, n), cfg.cluster, cfg.workload.target_load)
            s = summarize(run_episode(factory(), make_cluster(cfg, noise_seed=seed), w))
            rows.append(dict(scheduler=name, n_tasks=n, **s.as_dict()))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return f"task scaling {ds} seed{seed}"


def job_device_scaling(ds: str, seed: int) -> str:
    """Fig. 8: 100..1000 devices (75% edge / 25% cloud), workload scaled with the cluster."""
    torch.set_num_threads(1)
    out = RESULTS / "experiments" / f"device_scaling_{ds}_seed{seed}.csv"
    if out.exists():
        return f"skip {out}"
    rows = []
    for name, factory, cfg in _schedulers(ds, seed):
        sp = load_splits(cfg)
        for D in range(100, 1001, 100):
            ne, nc = int(D * 0.75), D - int(D * 0.75)
            cc = replace(cfg.cluster, n_edge=ne, n_cloud=nc)
            w = rescale_load(sp.test, cc, cfg.workload.target_load)
            cl = Cluster(cc, seed=0, noise_seed=seed)
            s = summarize(run_episode(factory(), cl, w))
            rows.append(dict(scheduler=name, devices=D, **s.as_dict()))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return f"device scaling {ds} seed{seed}"


def job_homogeneity(ds: str, seed: int) -> str:
    """Table 9: failure rate on a single-type (homogeneous) vs all-type (heterogeneous) workload."""
    torch.set_num_threads(1)
    out = RESULTS / "experiments" / f"homogeneity_{ds}_seed{seed}.csv"
    if out.exists():
        return f"skip {out}"
    rows = []
    for name, factory, cfg in _schedulers(ds, seed):
        sp = load_splits(cfg)
        top = int(np.bincount(sp.test_seen.ttype).argmax())
        homo = rescale_load(sp.test.subset(np.flatnonzero(sp.test.ttype == top)), cfg.cluster,
                            cfg.workload.target_load)
        hetero = rescale_load(sp.test, cfg.cluster, cfg.workload.target_load)
        for label, w in (("homogeneous", homo), ("heterogeneous", hetero)):
            s = summarize(run_episode(factory(), make_cluster(cfg, noise_seed=seed), w))
            rows.append(dict(scheduler=name, workload=label, failure_rate=100 - s.sla, **s.as_dict()))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return f"homogeneity {ds} seed{seed}"


def job_index_benchmark() -> str:
    """Retrieval cost of the B+-tree/iDistance index vs exhaustive scan, N = 100..50,000."""
    from caprf.index.bplustree import AdaptiveRetriever, IDistanceIndex, QueryStats
    out = RESULTS / "experiments" / "index_benchmark.csv"
    if out.exists():
        return f"skip {out}"
    sched, cfg = load_caprf("bitbrains", 0)
    rng = np.random.default_rng(0)
    rows = []
    for N in (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000):
        ne, nc = int(N * 0.75), N - int(N * 0.75)
        cl = Cluster(replace(ClusterConfig(), n_edge=ne, n_cloud=nc), seed=1)
        cl.core_free = np.where(np.isfinite(cl.core_free), rng.exponential(1.0, cl.core_free.shape), np.inf)
        emb = sched.net.embed_resources(torch.as_tensor(cl.descriptors())).numpy()
        sp = load_splits(cfg)
        cont = np.stack([sched._task_features(sp.test, i, update=False)[0] for i in range(200)])
        types = torch.as_tensor([sched.type_map.get(int(t), sched.n_emb) for t in sp.test.ttype[:200]])
        q = sched.net.embed_tasks(torch.as_tensor(cont), types).numpy()
        idx = IDistanceIndex(cfg.caprf.n_refs, cfg.caprf.btree_order)
        t = time.perf_counter(); idx.build(emb); build_s = time.perf_counter() - t
        k = cfg.caprf.k_max
        ret = AdaptiveRetriever(idx, cfg.caprf.k_min, k)
        for qq in q[:20]:
            ret.query(qq)  # warm up tau
        ret.stats = QueryStats()
        t = time.perf_counter()
        for qq in q:
            ret.query(qq)
        idx_ms = 1000 * (time.perf_counter() - t) / len(q)
        t = time.perf_counter()
        for qq in q:  # exhaustive scan, pure Python (same language level as the tree)
            d = [float(np.sqrt(((e - qq) ** 2).sum())) for e in emb] if N <= 5000 else None
            if d is None:
                break
        lin_py_ms = 1000 * (time.perf_counter() - t) / len(q) if N <= 5000 else np.nan
        t = time.perf_counter()
        for qq in q:  # exhaustive scan, vectorised NumPy
            d = np.linalg.norm(emb - qq, axis=1)
            np.argpartition(d, min(k, N - 1))[:k]
        lin_np_ms = 1000 * (time.perf_counter() - t) / len(q)
        upd = rng.integers(0, N, 200)
        t = time.perf_counter()
        for j in upd:
            idx.update(int(j), emb[j] + rng.normal(0, 0.01, emb.shape[1]))
        upd_ms = 1000 * (time.perf_counter() - t) / len(upd)
        rows.append(dict(N=N, btree_query_ms=idx_ms, linear_python_ms=lin_py_ms, linear_numpy_ms=lin_np_ms,
                         btree_update_ms=upd_ms, build_ms=1000 * build_s, tree_height=idx.tree.height(),
                         distance_evals_per_query=ret.stats.distance_evals / len(q),
                         frac_evaluated=ret.stats.distance_evals / len(q) / N))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return "index benchmark"


# ------------------------------------------------------------------------------- driver
def _run(pool, jobs, label):
    print(f"== {label}: {len(jobs)} jobs", flush=True)
    futs = {pool.submit(f, *a): (f.__name__, a) for f, a in jobs}
    for fu in as_completed(futs):
        try:
            print("  ", fu.result(), flush=True)
        except Exception as e:  # keep going; report at the end
            print("   FAILED", futs[fu], repr(e), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--skip-ablation", action="store_true", help="train only the full model")
    a = ap.parse_args()
    seeds = range(a.seeds)
    for ds in a.datasets:  # build/cached workloads once, before forking
        load_splits(dataset_config(ds))
    variants = [FULL] if a.skip_ablation else range(len(ABLATION))
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as pool:
        # full model first so the follow-up experiments can start early
        order = sorted(variants, key=lambda v: v != FULL)
        _run(pool, [(job_train, (ds, vi, s, a.steps)) for vi in order for ds in a.datasets for s in seeds]
             + [(job_heuristic, (ds, hi, s)) for ds in a.datasets for hi in range(len(HEURISTICS)) for s in seeds],
             "training + baselines")
        _run(pool, [(job_task_scaling, (ds, 0)) for ds in a.datasets]
             + [(job_device_scaling, ("kaggle", 0))]
             + [(job_homogeneity, (ds, s)) for ds in a.datasets for s in seeds]
             + [(job_index_benchmark, ())], "follow-up experiments")
    print(f"all done in {(time.time() - t0) / 60:.1f} min")
    (RESULTS / "run_info.json").write_text(json.dumps(vars(a), indent=2))


if __name__ == "__main__":
    main()
