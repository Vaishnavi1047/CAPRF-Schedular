"""Discrete-event cloud-edge cluster simulator (the paper's "custom simulator", Sec. 4.1).

Model, per node j:
  * `cores` CPU cores at `ghz`; a task needing k cores starts when k cores are free
    (list scheduling over per-core free times, non-preemptive) and runs cycles/ghz seconds.
  * a network link (bandwidth + propagation delay) the task's input data crosses first.
  * a small LRU pool of warm runtime containers, one per task type. A task whose type has
    no warm container pays the node's cold-start delay (50-200 ms, Table 2) plus energy.
    Containers stay warm for `keepalive_s` after last use. Prewarming starts one in advance.
  * power per active core, and a grid carbon intensity that varies over the day.

Logical resource isolation is assumed (Sec. 3.3): no interference between co-located tasks.
Predicted metrics (used by the schedulers) are noise-free; the executed task adds
log-normal runtime noise, so predictions are approximations as in the paper.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from caprf.config import ClusterConfig
from caprf.data.workload import Workload

NET_J_PER_MB = {0: 0.6, 1: 1.2}  # edge, cloud transfer energy (impl.)
DAY_S = 300.0                    # one simulated "day" of carbon variation (impl.)


@dataclass
class TaskRecord:
    task: int
    node: int
    arrival: float
    start: float
    finish: float
    latency: float
    wait: float
    exec_s: float
    cold_s: float
    energy_j: float
    carbon_g: float
    sla: bool
    warm_hit: bool


class Cluster:
    def __init__(self, cc: ClusterConfig, seed: int, n_edge: int | None = None, n_cloud: int | None = None,
                 noise_seed: int = 0):
        self.cc = cc
        rng = np.random.default_rng(seed + 7)
        ne = cc.n_edge if n_edge is None else n_edge
        nc = cc.n_cloud if n_cloud is None else n_cloud
        self.n = ne + nc
        self.is_cloud = np.r_[np.zeros(ne, bool), np.ones(nc, bool)]

        def draw(edge, cloud, integer=False):
            f = rng.integers if integer else rng.uniform
            e = f(edge[0], edge[1] + (1 if integer else 0), ne)
            c = f(cloud[0], cloud[1] + (1 if integer else 0), nc)
            return np.r_[e, c]

        self.cores = draw(cc.edge_cores, cc.cloud_cores, integer=True).astype(np.int64)
        self.mem = draw(cc.edge_mem_gb, cc.cloud_mem_gb)
        self.bw = draw(cc.edge_bw_mbps, cc.cloud_bw_mbps)
        self.ghz = draw(cc.edge_ghz, cc.cloud_ghz)
        self.prop = draw(cc.edge_prop_s, cc.cloud_prop_s)
        self.watt = draw(cc.edge_watt_per_core, cc.cloud_watt_per_core)
        self.carbon_base = draw(cc.edge_carbon, cc.cloud_carbon)
        self.carbon_phase = rng.uniform(0, 2 * np.pi, self.n)
        self.cold = rng.uniform(*cc.cold_start_s, self.n)
        self.slots = np.where(self.is_cloud, cc.cloud_container_slots, cc.edge_container_slots)
        self.net_j = np.where(self.is_cloud, NET_J_PER_MB[1], NET_J_PER_MB[0])
        self.total_cores = int(self.cores.sum())
        self.noise_seed = noise_seed
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.rng = np.random.default_rng(1000 + self.noise_seed)  # runtime noise, replayable per episode
        # per-core free times, padded with +inf for cores a node does not have
        self.core_free = np.full((self.n, int(self.cores.max())), np.inf)
        for j, c in enumerate(self.cores):
            self.core_free[j, :c] = 0.0
        # node -> OrderedDict(type -> [ready_time, expire_time]) in LRU order
        self.containers = [OrderedDict() for _ in range(self.n)]
        self.busy_core_s = np.zeros(self.n)
        self.prewarm_energy_j = 0.0
        self.prewarm_carbon_g = 0.0
        self.n_prewarms = 0
        self.now = 0.0

    def carbon(self, t: float, nodes=slice(None)) -> np.ndarray:
        amp = self.cc.carbon_diurnal_amp
        return self.carbon_base[nodes] * (1 + amp * np.sin(2 * np.pi * t / DAY_S + self.carbon_phase[nodes]))

    def kth_free(self, nodes: np.ndarray, k: int) -> np.ndarray:
        """Earliest time each node has k free cores (inf if it has fewer than k)."""
        k = int(k)
        if k > self.core_free.shape[1]:
            return np.full(len(nodes), np.inf)
        return np.partition(self.core_free[nodes], k - 1, axis=1)[:, k - 1]

    def backlog(self) -> np.ndarray:
        busy = np.where(np.isfinite(self.core_free), np.maximum(self.core_free - self.now, 0), 0)
        return busy.sum(1) / self.cores

    def warm_ready(self, j: int, ttype: int, t: float) -> float | None:
        """Time at which a container of ttype is usable on j for a task starting at t, or None."""
        c = self.containers[j].get(ttype)
        if c is None or c[1] < t:
            return None
        return c[0]

    # ------------------------------------------------------------------ prediction
    def predict(self, w: Workload, i: int, nodes: np.ndarray) -> dict:
        """Noise-free estimates of L_ij, E_ij, C_ij, O_ij (Sec. 3.3) for candidate nodes."""
        nodes = np.asarray(nodes)
        k = int(w.cores[i])
        ready = self.now + self.prop[nodes] + w.data_mb[i] * 8.0 / self.bw[nodes]
        free = self.kth_free(nodes, k)
        start = np.maximum(ready, free)
        cold = np.empty(len(nodes))
        warm = np.zeros(len(nodes), bool)
        for n_, j in enumerate(nodes):
            r = self.warm_ready(j, int(w.ttype[i]), start[n_])
            if r is None:
                cold[n_] = self.cold[j]
            else:
                warm[n_] = True
                cold[n_] = max(r - start[n_], 0.0)
        exec_s = w.cycles[i] / self.ghz[nodes]
        lat = start + cold + exec_s - self.now
        energy = self.watt[nodes] * k * exec_s + self.cc.cold_start_watt * cold + self.net_j[nodes] * w.data_mb[i]
        carbon = energy / 3.6e6 * self.carbon(self.now, nodes) * 1000.0
        feasible = k <= self.cores[nodes]
        lat = np.where(feasible, lat, np.inf)
        return dict(nodes=nodes, latency=lat, energy=energy, carbon=carbon, cold=cold,
                    warm=warm, wait=np.maximum(free - ready, 0.0), exec_s=exec_s, feasible=feasible)

    # ------------------------------------------------------------------ actions
    def _touch_container(self, j: int, ttype: int, ready: float, expire: float) -> None:
        pool = self.containers[j]
        if ttype in pool:
            pool[ttype][0] = min(pool[ttype][0], ready) if pool[ttype][1] >= ready else ready
            pool[ttype][1] = max(pool[ttype][1], expire)
            pool.move_to_end(ttype)
            return
        # drop expired containers, then evict LRU if still full
        for t in [t for t, c in pool.items() if c[1] < self.now]:
            del pool[t]
        while len(pool) >= self.slots[j]:
            pool.popitem(last=False)
        pool[ttype] = [ready, expire]

    def assign(self, w: Workload, i: int, j: int) -> TaskRecord:
        k = int(w.cores[i])
        ttype = int(w.ttype[i])
        ready = self.now + self.prop[j] + w.data_mb[i] * 8.0 / self.bw[j]
        cf = self.core_free[j]
        idx = np.argpartition(cf, k - 1)[:k]
        start = max(ready, float(cf[idx].max()))
        r = self.warm_ready(j, ttype, start)
        warm_hit = r is not None
        cold = (self.cold[j] * self.rng.lognormal(0, 0.1)) if r is None else max(r - start, 0.0)
        exec_s = w.cycles[i] / self.ghz[j] * self.rng.lognormal(0, 0.1)
        finish = start + cold + exec_s
        cf[idx] = finish
        self.busy_core_s[j] += k * (cold + exec_s)
        self._touch_container(j, ttype, start + cold, finish + self.cc.keepalive_s)
        energy = self.watt[j] * k * exec_s + self.cc.cold_start_watt * cold + self.net_j[j] * w.data_mb[i]
        carbon = energy / 3.6e6 * float(self.carbon(self.now, j)) * 1000.0
        lat = finish - self.now
        return TaskRecord(i, j, self.now, start, finish, lat, start - ready, exec_s, cold,
                          energy, carbon, lat <= w.deadline[i], warm_hit)

    def prewarm(self, j: int, ttype: int) -> bool:
        """Start a container of ttype on node j now (no-op if already warm or warming)."""
        c = self.containers[j].get(ttype)
        if c is not None and c[1] >= self.now:
            return False
        ready = self.now + self.cold[j]
        self._touch_container(j, ttype, ready, ready + self.cc.keepalive_s)
        e = self.cc.cold_start_watt * self.cold[j]
        self.prewarm_energy_j += e
        self.prewarm_carbon_g += e / 3.6e6 * float(self.carbon(self.now, j)) * 1000.0
        self.n_prewarms += 1
        return True

    # ------------------------------------------------------------------ descriptors
    def descriptors(self) -> np.ndarray:
        """Raw resource descriptor (a_j, e_j, l_j, gamma_j, is_cloud) of Eq. (7), scaled to ~[0, 1]."""
        bl = self.backlog()
        a = np.exp(-bl / 1.0)                                  # availability
        e = (self.ghz / self.watt) / 0.8                       # GHz per watt (energy efficiency)
        l = (self.prop + 1.0 * 8 / self.bw + 1.5 / self.ghz + bl) / 2.0  # latency for a standard task
        g = self.carbon(self.now) / 0.5                        # carbon intensity
        return np.column_stack([a, e, l, g, self.is_cloud.astype(float)]).astype(np.float32)
