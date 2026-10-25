"""Shared helpers: dataset presets, training loop, evaluation."""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from caprf.config import Config, Variant
from caprf.data.workload import Splits, make_splits, rescale_load
from caprf.schedulers.caprf import CAPRFScheduler
from caprf.sim.cluster import Cluster
from caprf.sim.runner import run_episode, summarize

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("CAPRF_RESULTS", ROOT / "results"))
EPISODE_LEN = 4000
VAL_FRAC = 0.15      # tail of the training period held out for checkpoint selection
VAL_EVERY = 5        # episodes between validation runs


def dataset_config(name: str, seed: int = 0, variant: Variant | None = None) -> Config:
    cfg = Config(seed=seed)
    cfg.workload.dataset = name
    if name == "kaggle":  # 20k tasks, no timestamps, smoother arrivals -> higher load to stress it
        cfg.workload = replace(cfg.workload, n_tasks=20_000, n_types=6, unseen_types=1, target_load=0.9)
    else:
        cfg.workload = replace(cfg.workload, n_tasks=50_000, n_types=8, unseen_types=2, target_load=0.72)
    if variant is not None:
        cfg.variant = variant
    return cfg


def load_splits(cfg: Config) -> Splits:
    # the workload (task stream + held-out types) depends on the data seed only; RL seeds vary separately
    return make_splits(cfg.workload, cfg.cluster, seed=0)


def make_cluster(cfg: Config, **kw) -> Cluster:
    return Cluster(cfg.cluster, seed=0, **kw)


def train_caprf(cfg: Config, splits: Splits, steps: int | None = None, verbose: bool = True,
                log_path: Path | None = None) -> tuple[CAPRFScheduler, pd.DataFrame]:
    torch.set_num_threads(1)
    steps = steps or cfg.ppo.train_steps
    seen = sorted(set(np.unique(splits.train.ttype).tolist()))
    sched = CAPRFScheduler(cfg, n_types=len(splits.full.type_names), seen_types=seen)
    cluster = make_cluster(cfg)
    rng = np.random.default_rng(cfg.seed)
    n_val = int(len(splits.train) * VAL_FRAC)
    train = splits.train.window(0, len(splits.train) - n_val)
    val = rescale_load(splits.train.window(len(splits.train) - n_val, n_val), cfg.cluster,
                       cfg.workload.target_load)
    best = (-np.inf, None, 0)
    done, ep, rows = 0, 0, []
    t0 = time.time()
    while done < steps:
        start = int(rng.integers(0, max(len(train) - EPISODE_LEN, 1)))
        # domain randomisation over load: train under lighter and heavier traffic than the target
        load = float(rng.uniform(cfg.workload.target_load - 0.15, cfg.workload.target_load + 0.1))
        w = rescale_load(train.window(start, EPISODE_LEN), cfg.cluster, load)
        df = run_episode(sched, cluster, w, train=True)
        s = summarize(df)
        done += len(w)
        ep += 1
        upd = sched.learner.log[-1] if sched.learner.log else {}
        row = dict(episode=ep, steps=done, latency_ms=s.latency_ms, sla=s.sla, carbon_g=s.carbon_g,
                   energy_kj=s.energy_kj, warm_hit=s.warm_hit, reward=sched.episode_log[-1]["reward"],
                   entropy=upd.get("entropy", np.nan), elapsed_s=time.time() - t0)
        if ep % VAL_EVERY == 0 or done >= steps:
            ev = sched.eval_copy()
            vdf = run_episode(ev, cluster, val, train=False)
            row["val_reward"] = ev.episode_log[-1]["reward"] / len(val)
            row["val_sla"] = 100 * vdf.sla.mean()
            if row["val_reward"] > best[0]:
                best = (row["val_reward"], sched.snapshot(), ep)
        rows.append(row)
        if verbose:
            print(f"[{cfg.workload.dataset}|{cfg.variant.name}|seed{cfg.seed}] ep {ep:3d} steps {done:7d} "
                  f"lat {s.latency_ms:7.0f}ms sla {s.sla:5.1f}% carbon {s.carbon_g:6.1f}g "
                  f"warm {s.warm_hit:4.1f}% R {row['reward']:8.1f} H {row['entropy']:.2f} "
                  f"({row['elapsed_s']:.0f}s)", flush=True)
        if log_path is not None:
            pd.DataFrame(rows).to_csv(log_path, index=False)
    if best[1] is not None:  # keep the checkpoint with the best validation reward
        sched.restore(best[1])
        sched.best_episode = best[2]
    return sched, pd.DataFrame(rows)


def evaluate(sched, cfg: Config, w, cluster: Cluster | None = None) -> pd.DataFrame:
    cluster = cluster or make_cluster(cfg)
    if isinstance(sched, CAPRFScheduler):
        sched = sched.eval_copy()
    return run_episode(sched, cluster, w, train=False)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
