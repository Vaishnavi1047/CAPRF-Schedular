"""Turn the two raw datasets into task streams with the attributes CAPRF++ needs
(Sec. 3.1): deadline d_i, compute intensity c_i, priority p_i, sustainability preference s_i,
task type t_i, plus the resource demands of Table 2 (cores, memory, data size).

Mapping choices (the paper does not spell these out, so they are documented here):
  * compute intensity  -> rescaled (log min-max) into Table 2's 0.1-5 billion cycles
  * cores / memory / data size -> clipped or rescaled into Table 2's ranges
  * task type          -> k-means workload class (Bitbrains: per VM usage profile,
                          Kaggle: per task resource profile)
  * arrivals           -> Bitbrains: real trace timestamps, 30 days compressed into the
                          simulation horizon (keeps the diurnal/weekly shape);
                          Kaggle: has no timestamps, so a Markov-modulated Poisson process
                          with phase-dependent type mix is synthesised
  * priority           -> Kaggle: given column; Bitbrains: per-VM draw (not in the trace)
  * sustainability s_i -> per-type mean + noise (not in either dataset)
  * deadline           -> priority-dependent slack x reference execution time
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from caprf.config import ClusterConfig, WorkloadConfig

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FIELDS = ("arrival", "cycles", "cores", "mem_gb", "data_mb", "priority",
          "deadline", "sustain", "ttype")
SLACK = {1: (1.6, 2.4), 2: (2.4, 3.6), 3: (3.6, 5.5)}  # priority -> deadline slack range


@dataclass
class Workload:
    arrival: np.ndarray   # s, sorted
    cycles: np.ndarray    # billion cycles per thread
    cores: np.ndarray     # int, 2..10
    mem_gb: np.ndarray
    data_mb: np.ndarray
    priority: np.ndarray  # 1 = high, 2 = medium, 3 = low
    deadline: np.ndarray  # relative deadline, s
    sustain: np.ndarray   # 0..1, how strongly the task prefers green execution
    ttype: np.ndarray     # int workload class
    type_names: list = field(default_factory=list)
    unseen: set = field(default_factory=set)  # type ids never shown during training
    name: str = ""

    def __len__(self) -> int:
        return len(self.arrival)

    def subset(self, idx: np.ndarray, name: str | None = None, rebase: bool = True) -> "Workload":
        idx = np.sort(np.asarray(idx))
        kw = {f: getattr(self, f)[idx].copy() for f in FIELDS}
        if rebase and len(idx):
            kw["arrival"] = kw["arrival"] - kw["arrival"][0]
        return Workload(**kw, type_names=self.type_names, unseen=self.unseen,
                        name=name or self.name)

    def window(self, start: int, n: int) -> "Workload":
        return self.subset(np.arange(start, min(start + n, len(self))))

    def is_unseen(self) -> np.ndarray:
        return np.isin(self.ttype, list(self.unseen))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f: getattr(self, f) for f in FIELDS},
                            type_names=np.array(self.type_names), unseen=np.array(sorted(self.unseen)),
                            name=np.array(self.name))

    @staticmethod
    def load(path: Path) -> "Workload":
        z = np.load(path, allow_pickle=False)
        return Workload(**{f: z[f] for f in FIELDS}, type_names=list(z["type_names"]),
                        unseen=set(int(u) for u in z["unseen"]), name=str(z["name"]))


def _log_minmax(x: np.ndarray, lo: float, hi: float, q=(0.01, 0.99)) -> np.ndarray:
    lx = np.log1p(np.maximum(x, 0))
    a, b = np.quantile(lx, q)
    return lo + (hi - lo) * np.clip((lx - a) / max(b - a, 1e-9), 0, 1)


def reference_time(cycles, data_mb) -> np.ndarray:
    """Execution time of a task on a 'typical' node with no queueing (deadline anchor)."""
    return 0.02 + data_mb * 8 / 100.0 + cycles / 2.5 + 0.1


def _deadlines(rng, cycles, data_mb, priority) -> np.ndarray:
    lo = np.array([SLACK[int(p)][0] for p in priority])
    hi = np.array([SLACK[int(p)][1] for p in priority])
    return reference_time(cycles, data_mb) * rng.uniform(lo, hi)


def _sustain(rng, ttype, n_types) -> np.ndarray:
    means = rng.uniform(0.2, 0.8, n_types)
    return np.clip(means[ttype] + rng.normal(0, 0.12, len(ttype)), 0, 1)


def calibrate_horizon(cycles, cores, cluster: ClusterConfig, target_load: float) -> float:
    """Arrival horizon such that mean cluster utilisation equals target_load."""
    mean_ghz = (cluster.n_edge * np.mean(cluster.edge_ghz) * np.mean(cluster.edge_cores)
                + cluster.n_cloud * np.mean(cluster.cloud_ghz) * np.mean(cluster.cloud_cores))
    total_cores = cluster.n_edge * np.mean(cluster.edge_cores) + cluster.n_cloud * np.mean(cluster.cloud_cores)
    ghz = mean_ghz / total_cores
    core_seconds = float(np.sum(cores * cycles / ghz))
    return core_seconds / (target_load * total_cores)


def rescale_load(w: Workload, cluster: ClusterConfig, target_load: float) -> Workload:
    """Stretch/compress arrival times so the workload drives the cluster at target_load."""
    h = calibrate_horizon(w.cycles, w.cores, cluster, target_load)
    span = max(w.arrival[-1] - w.arrival[0], 1e-9)
    out = w.subset(np.arange(len(w)))
    out.arrival = (out.arrival - out.arrival[0]) * (h / span)
    return out


# ---------------------------------------------------------------- Bitbrains (Dataset 1)
BB_COLS = ["ts", "cores", "cap_mhz", "cpu_mhz", "cpu_pct", "mem_cap_kb", "mem_kb",
           "disk_r", "disk_w", "net_rx", "net_tx"]


def _load_bitbrains_raw(n_vms: int) -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / "bitbrains" / "fastStorage" / "*.csv")),
                   key=lambda f: int(Path(f).stem))[:n_vms]
    if not files:
        raise FileNotFoundError("Bitbrains trace missing - run: python scripts/download_data.py")
    parts = []
    for vm, f in enumerate(files):
        d = pd.read_csv(f, sep=r";\s*|,", engine="python") if vm == 0 else pd.read_csv(f)
        d = d.iloc[:, :11]
        d.columns = BB_COLS
        d["vm"] = vm
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def build_bitbrains(wc: WorkloadConfig, cc: ClusterConfig, seed: int) -> Workload:
    rng = np.random.default_rng(seed)
    df = _load_bitbrains_raw(wc.bitbrains_vms)
    df = df[df.cpu_pct > 1.0].copy()  # an "active" 5-minute sample becomes one task
    df["io"] = df.disk_r + df.disk_w + df.net_rx + df.net_tx

    # workload classes: k-means over each VM's usage profile
    prof = df.groupby("vm").agg(cpu=("cpu_pct", "mean"), cpu_sd=("cpu_pct", "std"),
                                mem=("mem_kb", "mean"), disk=("disk_w", "mean"),
                                net=("net_rx", "mean"), cores=("cores", "median")).fillna(0)
    X = np.column_stack([prof.cpu, prof.cpu_sd, np.log1p(prof.mem), np.log1p(prof.disk),
                         np.log1p(prof.net), prof.cores])
    km = KMeans(wc.n_types, n_init=10, random_state=seed).fit(StandardScaler().fit_transform(X))
    vm_type = dict(zip(prof.index, km.labels_))
    vm_prio = dict(zip(prof.index, rng.choice([1, 2, 3], len(prof), p=[0.3, 0.4, 0.3])))

    df = df.sample(n=min(wc.n_tasks, len(df)), random_state=seed).sort_values("ts")
    ttype = df.vm.map(vm_type).to_numpy().astype(np.int64)
    cycles = _log_minmax(df.cpu_mhz.to_numpy() * 300.0, 0.1, 5.0)
    cores = np.clip(df.cores.to_numpy(), 2, 10).astype(np.int64)
    mem = np.clip(df.mem_kb.to_numpy() / 1e6, 1, 8)
    data = _log_minmax(df.io.to_numpy(), 0.1, 5.0)
    prio = df.vm.map(vm_prio).to_numpy().astype(np.int64)

    horizon = calibrate_horizon(cycles, cores, cc, wc.target_load)
    ts = df.ts.to_numpy().astype(float)
    arrival = (ts - ts[0]) / (ts[-1] - ts[0]) * horizon
    arrival = np.sort(arrival + rng.uniform(0, horizon / len(ts), len(ts)))

    names = _type_names(km.cluster_centers_, ["cpu", "cpu-burst", "mem", "disk", "net", "cores"])
    return Workload(arrival, cycles, cores, mem, data, prio,
                    _deadlines(rng, cycles, data, prio), _sustain(rng, ttype, wc.n_types),
                    ttype, names, name="bitbrains")


# ---------------------------------------------------------------- Kaggle (Dataset 2)
def build_kaggle(wc: WorkloadConfig, cc: ClusterConfig, seed: int) -> Workload:
    rng = np.random.default_rng(seed)
    path = RAW / "kaggle" / "cloud_task_scheduling_dataset.csv"
    if not path.exists():
        raise FileNotFoundError("Kaggle dataset missing - run: python scripts/download_data.py")
    df = pd.read_csv(path)
    df.columns = ["task_id", "cpu", "ram", "disk", "net", "priority", "vm", "exec", "target"]
    df = df.head(wc.n_tasks).reset_index(drop=True)

    X = df[["cpu", "ram", "disk", "net"]].to_numpy(float)
    km = KMeans(wc.n_types, n_init=10, random_state=seed).fit(StandardScaler().fit_transform(X))
    ttype_all = km.labels_.astype(np.int64)

    # Markov-modulated Poisson arrivals: each phase has its own intensity and type mix,
    # which gives the temporal locality that arrival prediction / prewarming relies on.
    n = len(df)
    order, arrivals, t = [], [], 0.0
    pools = {k: list(rng.permutation(np.flatnonzero(ttype_all == k))) for k in range(wc.n_types)}
    while len(order) < n:
        dur = rng.exponential(3.0)
        rate = rng.choice([0.5, 1.0, 1.8])
        mix = rng.dirichlet(np.full(wc.n_types, 0.4))
        end = t + dur
        while t < end and len(order) < n:
            t += rng.exponential(1.0 / rate)
            avail = np.array([len(pools[k]) > 0 for k in range(wc.n_types)], float)
            p = mix * avail
            p = p / p.sum() if p.sum() > 0 else avail / avail.sum()
            k = rng.choice(wc.n_types, p=p)
            order.append(pools[k].pop())
            arrivals.append(t)
    df = df.iloc[order].reset_index(drop=True)
    ttype = ttype_all[order]

    cycles = _log_minmax(df.exec.to_numpy() * df.cpu.to_numpy() / 100.0, 0.1, 5.0, q=(0.0, 1.0))
    cores = np.clip(2 + np.round(df.cpu.to_numpy() / 100.0 * 8), 2, 10).astype(np.int64)
    mem = np.clip(df.ram.to_numpy() / 1024.0, 1, 8)
    data = 0.1 + 4.9 * (df.net.to_numpy() - df.net.min()) / max(df.net.max() - df.net.min(), 1)
    prio = df.priority.to_numpy().astype(np.int64)

    horizon = calibrate_horizon(cycles, cores, cc, wc.target_load)
    arrival = np.asarray(arrivals)
    arrival = arrival / arrival[-1] * horizon

    names = _type_names(km.cluster_centers_, ["cpu", "ram", "disk", "net"])
    return Workload(arrival, cycles, cores, mem, data, prio,
                    _deadlines(rng, cycles, data, prio), _sustain(rng, ttype, wc.n_types),
                    ttype, names, name="kaggle")


def _type_names(centers: np.ndarray, feats: list[str]) -> list[str]:
    names = []
    for c in centers:
        top = np.argsort(-c)[:2]
        tag = "+".join(f"{feats[i]}{'↑' if c[i] > 0 else '↓'}" for i in top)
        names.append(f"T{len(names)}:{tag}")
    return names


# ---------------------------------------------------------------- splits
@dataclass
class Splits:
    full: Workload
    train: Workload        # chronological prefix, unseen types removed
    test: Workload         # chronological suffix, all types (seen + unseen)
    test_seen: Workload    # test window, seen types only


def make_splits(wc: WorkloadConfig, cc: ClusterConfig, seed: int, cache: bool = True) -> Splits:
    tag = f"{wc.dataset}_n{wc.n_tasks}_k{wc.n_types}_u{wc.unseen_types}_l{wc.target_load}_s{seed}"
    path = PROCESSED / f"{tag}.npz"
    if cache and path.exists():
        full = Workload.load(path)
    else:
        build = build_bitbrains if wc.dataset == "bitbrains" else build_kaggle
        full = build(wc, cc, seed)
        rng = np.random.default_rng(seed + 991)
        full.unseen = set(int(u) for u in rng.choice(wc.n_types, wc.unseen_types, replace=False))
        if cache:
            full.save(path)
    cut = int(len(full) * wc.train_frac)
    unseen = full.is_unseen()
    idx = np.arange(len(full))
    train = full.subset(idx[(idx < cut) & ~unseen], name=f"{full.name}-train")
    test = full.subset(idx[idx >= cut], name=f"{full.name}-test")
    test_seen = full.subset(idx[(idx >= cut) & ~unseen], name=f"{full.name}-test-seen")
    return Splits(full, train, test, test_seen)
