"""Episode runner + the evaluation metrics of Sec. 4.2."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from caprf.data.workload import Workload
from caprf.sim.cluster import Cluster, TaskRecord


class Scheduler:
    name = "scheduler"

    def begin(self, cluster: Cluster, w: Workload, train: bool = False) -> None:
        pass

    def decide(self, cluster: Cluster, w: Workload, i: int) -> int:
        raise NotImplementedError

    def observe(self, cluster: Cluster, w: Workload, i: int, rec: TaskRecord) -> None:
        pass

    def end(self, cluster: Cluster, w: Workload) -> None:
        pass


def run_episode(sched: Scheduler, cluster: Cluster, w: Workload, train: bool = False,
                progress=None) -> pd.DataFrame:
    cluster.reset()
    sched.begin(cluster, w, train)
    recs = []
    for i in range(len(w)):
        cluster.now = float(w.arrival[i])
        j = sched.decide(cluster, w, i)
        rec = cluster.assign(w, i, j)
        sched.observe(cluster, w, i, rec)
        recs.append(rec)
        if progress is not None and i % 500 == 0:
            progress(i, len(w))
    sched.end(cluster, w)
    df = pd.DataFrame([r.__dict__ for r in recs])
    df["ttype"] = w.ttype
    df["unseen"] = w.is_unseen()
    df["deadline"] = w.deadline
    df["is_cloud"] = cluster.is_cloud[df.node.to_numpy()]
    df.attrs["prewarm_energy_j"] = cluster.prewarm_energy_j
    df.attrs["prewarm_carbon_g"] = cluster.prewarm_carbon_g
    df.attrs["n_prewarms"] = cluster.n_prewarms
    df.attrs["busy_core_s"] = cluster.busy_core_s.copy()
    df.attrs["cores"] = cluster.cores.copy()
    return df


@dataclass
class Summary:
    latency_ms: float
    p95_latency_ms: float
    sla: float                 # % tasks finishing within deadline
    energy_kj: float           # task energy + prewarming energy
    carbon_g: float
    cold_start_s: float        # total cold-start delay paid by tasks
    warm_hit: float            # % tasks that found a warm container (executor reuse rate)
    makespan_s: float
    utilization: float         # % busy core time of nodes that ran work
    sla_seen: float
    sla_unseen: float
    generalization: float      # 100 * SLA(unseen types) / SLA(seen types), capped at 100
    cloud_share: float
    n_prewarms: int

    def as_dict(self):
        return self.__dict__.copy()


def summarize(df: pd.DataFrame) -> Summary:
    makespan = float(df.finish.max() - df.arrival.min())
    busy, cores = df.attrs["busy_core_s"], df.attrs["cores"]
    used = busy > 0
    util = 100.0 * busy[used].sum() / max(cores[used].sum() * makespan, 1e-9)
    seen, unseen = df[~df.unseen], df[df.unseen]
    sla_seen = 100.0 * seen.sla.mean() if len(seen) else np.nan
    sla_unseen = 100.0 * unseen.sla.mean() if len(unseen) else np.nan
    gen = min(100.0, 100.0 * sla_unseen / sla_seen) if len(unseen) and sla_seen > 0 else np.nan
    return Summary(
        latency_ms=1000 * df.latency.mean(),
        p95_latency_ms=1000 * df.latency.quantile(0.95),
        sla=100.0 * df.sla.mean(),
        energy_kj=(df.energy_j.sum() + df.attrs["prewarm_energy_j"]) / 1000,
        carbon_g=df.carbon_g.sum() + df.attrs["prewarm_carbon_g"],
        cold_start_s=float(df.cold_s.sum()),
        warm_hit=100.0 * df.warm_hit.mean(),
        makespan_s=makespan,
        utilization=util,
        sla_seen=sla_seen,
        sla_unseen=sla_unseen,
        generalization=gen,
        cloud_share=100.0 * df.is_cloud.mean(),
        n_prewarms=int(df.attrs["n_prewarms"]),
    )


def reductions(s: Summary, ref: Summary) -> dict:
    """Carbon-footprint and cold-start-overhead reduction rates vs a reference scheduler
    with no sustainability objective and no prewarming (Sec. 4.3.1)."""
    return dict(carbon_reduction=100 * (1 - s.carbon_g / ref.carbon_g),
                cold_start_reduction=100 * (1 - s.cold_start_s / max(ref.cold_start_s, 1e-9)),
                energy_reduction=100 * (1 - s.energy_kj / ref.energy_kj))


def per_interval(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Split tasks into n consecutive execution intervals (Figs. 5/6)."""
    g = np.arange(len(df)) * n // len(df)
    out = df.groupby(g).agg(latency_ms=("latency", lambda x: 1000 * x.mean()),
                            sla=("sla", lambda x: 100 * x.mean()),
                            carbon_g=("carbon_g", "sum"), energy_kj=("energy_j", lambda x: x.sum() / 1000),
                            warm_hit=("warm_hit", lambda x: 100 * x.mean()),
                            cold_ms=("cold_s", lambda x: 1000 * x.mean()),
                            start=("arrival", "min"), end=("finish", "max"))
    out["makespan_s"] = out.end - out.start
    out.index.name = "interval"
    return out
