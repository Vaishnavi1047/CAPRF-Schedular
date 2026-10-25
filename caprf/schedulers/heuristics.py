"""Non-learning reference schedulers.

The paper compares against six published systems (Ma et al., OTSA-PPO, TF-DDRL, Chen et al.,
MTGPER, ESDN) whose code is not public. Instead we compare against standard, reproducible
schedulers covering the same design space (the classical FCFS/RR/min-completion family the
paper's introduction discusses, a carbon-greedy policy, and Eq. (19) applied greedily), plus
a vanilla-PPO DRL scheduler (the ablation "Baseline", see caprf.schedulers.caprf).
All see the full resource pool and the same noise-free predictions as CAPRF++.
"""
from __future__ import annotations

import numpy as np

from caprf.sim.runner import Scheduler


class RandomScheduler(Scheduler):
    name = "Random"

    def begin(self, cluster, w, train=False):
        self.rng = np.random.default_rng(0)

    def decide(self, cluster, w, i):
        ok = np.flatnonzero(cluster.cores >= w.cores[i])
        return int(self.rng.choice(ok))


class RoundRobinScheduler(Scheduler):
    name = "Round-Robin"

    def begin(self, cluster, w, train=False):
        self.ptr = 0

    def decide(self, cluster, w, i):
        while True:
            j = self.ptr % cluster.n
            self.ptr += 1
            if cluster.cores[j] >= w.cores[i]:
                return j


class _PredictiveScheduler(Scheduler):
    def begin(self, cluster, w, train=False):
        self.all = np.arange(cluster.n)


class LeastLoadedScheduler(_PredictiveScheduler):
    """Earliest time k cores are free (load balancing)."""
    name = "Least-Loaded"

    def decide(self, cluster, w, i):
        free = cluster.kth_free(self.all, int(w.cores[i]))
        return int(np.argmin(free))


class MinCompletionScheduler(_PredictiveScheduler):
    """Online Min-Min / MCT: pick the node with the smallest predicted completion time."""
    name = "Greedy-MCT"

    def decide(self, cluster, w, i):
        return int(np.argmin(cluster.predict(w, i, self.all)["latency"]))


class GreenGreedyScheduler(_PredictiveScheduler):
    """Lowest predicted carbon among nodes that meet the deadline, else fastest."""
    name = "Green-Greedy"

    def decide(self, cluster, w, i):
        p = cluster.predict(w, i, self.all)
        ok = p["latency"] <= w.deadline[i]
        if ok.any():
            return int(self.all[ok][np.argmin(p["carbon"][ok])])
        return int(np.argmin(p["latency"]))


class WTCGreedyScheduler(_PredictiveScheduler):
    """Eq. (19) without learning: argmin of the weighted tradeoff score over all nodes."""
    name = "WTC-Greedy"

    def __init__(self, weights=(0.4, 0.3, 0.2, 0.1)):
        self.wt = weights

    def decide(self, cluster, w, i):
        p = cluster.predict(w, i, self.all)
        lat = np.where(np.isfinite(p["latency"]), p["latency"], 1e9)
        a, b, d, r = self.wt
        omega = (a * lat / np.median(lat) + b * p["energy"] / np.median(p["energy"])
                 + d * p["carbon"] / np.median(p["carbon"]) + r * p["cold"] / 0.2)
        omega = omega + 10.0 * (lat > w.deadline[i])
        return int(np.argmin(omega))


HEURISTICS = [RandomScheduler, RoundRobinScheduler, LeastLoadedScheduler, MinCompletionScheduler,
              GreenGreedyScheduler, WTCGreedyScheduler]
