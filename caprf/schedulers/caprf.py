"""CAPRF++ scheduler: the Actor of Algorithm 1 / the pipeline of Algorithm 3.

Per arriving task T_i:
  1. Context-aware feature construction  x_t -> z_t                          (Sec. 3.1)
  2. Candidate retrieval: B+-tree/iDistance range query ||z_t - z_r|| <= tau  (Sec. 3.2)
  3. Weighted Tradeoff Controller: Omega_ij, feasible set C_i                 (Sec. 3.3)
  4. PPO policy picks R*_j from C_i                                           (Sec. 3.4)
  5. Predictive prewarming on top-ranked candidates if lambda_i*Cold >= T_warm (Sec. 3.5)
  6. Assign, observe reward (Eq. 13), learn (Algorithm 2)

Every module can be switched off (config.Variant) for the ablation study.
"""
from __future__ import annotations

import copy
from collections import deque
from pathlib import Path

import numpy as np
import torch

from caprf.agent.ppo import PPOLearner
from caprf.config import Config
from caprf.data.workload import Workload
from caprf.index.bplustree import AdaptiveRetriever, BPlusTree, IDistanceIndex
from caprf.models.networks import ArrivalHead, PolicyValueNet, RunningNorm
from caprf.sim.cluster import Cluster, TaskRecord
from caprf.sim.runner import Scheduler

K_EXPLORE = 4  # random extra candidates during training only (keeps retrieval from collapsing)


class RunningMinMax:
    """Min-max scaling of reward terms using robust bounds from a sliding window (Sec. 3.4)."""

    def __init__(self, window: int = 5000):
        self.buf = deque(maxlen=window)
        self.lo, self.hi, self.k = 0.0, 1.0, 0

    def __call__(self, x: float) -> float:
        self.buf.append(x)
        self.k += 1
        if self.k % 250 == 1 or self.k < 50:
            a = np.asarray(self.buf)
            self.lo, self.hi = float(np.quantile(a, 0.01)), float(np.quantile(a, 0.99))
        return float(np.clip((x - self.lo) / max(self.hi - self.lo, 1e-9), 0, 1))


class EMAScale:
    def __init__(self, init: float):
        self.v = init

    def __call__(self, x: np.ndarray) -> float:
        m = float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else self.v
        self.v = 0.99 * self.v + 0.01 * m
        return self.v


class CAPRFScheduler(Scheduler):
    def __init__(self, cfg: Config, n_types: int, seen_types: list[int]):
        self.cfg = cfg
        self.v = cfg.variant
        self.name = cfg.variant.name
        cp = cfg.caprf
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.K = cp.k_max
        # type ids seen in training map to embedding rows; anything else -> UNK
        self.type_map = {t: k for k, t in enumerate(sorted(seen_types))}
        self.n_emb = len(self.type_map)
        self.net = PolicyValueNet(self.n_emb, cp.embed_dim, cp.type_embed_dim, cfg.ppo.hidden,
                                  context=self.v.context_encoding)
        self.arr_head = ArrivalHead(cp.embed_dim)
        self.learner = PPOLearner(self.net, self.arr_head, cfg.ppo, self.K, self.v.trajectory_buffer, cfg.seed)
        self.norm = RunningNorm(5)
        self.scales = dict(L=EMAScale(1.0), E=EMAScale(40.0), C=EMAScale(3e-3))
        self.rmm = {k: RunningMinMax() for k in "LECO"}
        self.train = False
        self.episode_log: list[dict] = []
        self.decision_times: list[float] = []

    # ------------------------------------------------------------------ episode hooks
    def begin(self, cluster: Cluster, w: Workload, train: bool = False) -> None:
        self.train = train
        self.net.train(train)
        cp = self.cfg.caprf
        self.index = IDistanceIndex(cp.n_refs, cp.btree_order, seed=self.cfg.seed)
        self.retriever = AdaptiveRetriever(self.index, cp.k_min, cp.k_max - (K_EXPLORE if train else 0))
        self._rebuild_index(cluster)
        # availability index: B+-tree keyed on each node's mean absolute core-free time. That
        # order does not change as the clock advances, so only the node that just received a
        # task needs re-keying (O(log N)).
        self.avail_key = {j: (float(cluster.core_free[j, : cluster.cores[j]].mean()), j) for j in range(cluster.n)}
        self.avail_tree = BPlusTree.bulk_load([(k, j) for j, k in self.avail_key.items()], cp.btree_order)
        self.since_refresh = 0
        self.type_count = np.zeros(int(w.ttype.max()) + 1 if len(w) else 1)
        self.total_count = 0.0
        self.last_t = 0.0
        self.mean_rate = len(w) / max(w.arrival[-1], 1e-9)
        self.pending = None
        self.ep_reward = 0.0
        self.prewarm_events = 0
        if train:  # supervision for lambda_i: does another task of this type arrive within the window?
            win = cp.arrival_window_s
            lab = np.zeros(len(w), np.float32)
            for t in np.unique(w.ttype):
                idx = np.flatnonzero(w.ttype == t)
                a = w.arrival[idx]
                nxt = np.r_[a[1:], np.inf]
                lab[idx] = (nxt - a) <= win
            self.arr_labels = lab

    def end(self, cluster: Cluster, w: Workload) -> None:
        if self.pending is not None:
            self._finish_pending(done=True)
        self.episode_log.append(dict(reward=self.ep_reward, prewarms=self.prewarm_events))

    # ------------------------------------------------------------------ features
    def _task_features(self, w: Workload, i: int, update: bool = True):
        raw = np.array([w.deadline[i], w.cycles[i], w.priority[i], w.sustain[i], w.cores[i]], float)
        if self.v.context_encoding:
            cont = self.norm(raw, update)  # Eq. (1)
        else:
            cont = raw / np.array([10.0, 5.0, 3.0, 1.0, 10.0])
        t = self.type_map.get(int(w.ttype[i]), self.n_emb)
        if self.train and self.v.context_encoding and self.rng.random() < self.cfg.caprf.unk_dropout:
            t = self.n_emb
        return cont.astype(np.float32), t

    def _rebuild_index(self, cluster: Cluster) -> None:
        self.desc = cluster.descriptors()
        self.res_emb = self.net.embed_resources(torch.as_tensor(self.desc)).numpy()
        if self.v.btree_index:
            self.index.build(self.res_emb)

    def _arrival_extra(self, w: Workload, i: int) -> np.ndarray:
        """Decayed per-type and total arrival counts over the prediction window (context)."""
        win = self.cfg.caprf.arrival_window_s
        t = float(w.arrival[i])
        decay = np.exp(-(t - self.last_t) / win)
        self.type_count *= decay
        self.total_count *= decay
        self.last_t = t
        k = int(w.ttype[i])
        self.type_count[k] += 1
        self.total_count += 1
        return np.array([np.log1p(self.type_count[k]), np.log1p(self.total_count) / 3], np.float32)

    # ------------------------------------------------------------------ WTC (Sec. 3.3)
    def _tradeoff(self, pred: dict, w: Workload, i: int):
        cp = self.cfg.caprf
        lat = np.where(np.isfinite(pred["latency"]), pred["latency"], 1e3)
        sL, sE, sC = self.scales["L"](lat), self.scales["E"](pred["energy"]), self.scales["C"](pred["carbon"])
        Ln, En, Cn, On = lat / sL, pred["energy"] / sE, pred["carbon"] / sC, pred["cold"] / 0.2
        omega = cp.alpha * Ln + cp.beta * En + cp.delta * Cn + cp.rho * On       # Eq. (10)
        dl = w.deadline[i]
        pair = np.column_stack([Ln, En, Cn, On, omega, pred["warm"], pred["wait"] / sL,
                                np.minimum(lat / dl, 3.0), self.desc[pred["nodes"], 4]])
        return omega, np.clip(pair, 0, 10).astype(np.float32), lat

    def _feasible(self, omega: np.ndarray, lat: np.ndarray, deadline: float, ok: np.ndarray) -> np.ndarray:
        """C_i = {R_j : Omega_ij <= Omega_min + Delta}, restricted to SLA-feasible nodes when any."""
        sla_ok = ok & (lat <= self.cfg.caprf.sla_margin * deadline)
        pool = sla_ok if sla_ok.any() else ok
        om = np.where(pool, omega, np.inf)
        keep = om <= om.min() + self.cfg.caprf.wtc_margin
        if keep.sum() < 3:  # always leave the agent some choice
            keep[np.argsort(om)[: min(3, int(pool.sum()))]] = True
        return np.flatnonzero(keep & pool)

    # ------------------------------------------------------------------ decision
    def decide(self, cluster: Cluster, w: Workload, i: int) -> int:
        import time
        t0 = time.perf_counter()
        cp = self.cfg.caprf
        cont, t = self._task_features(w, i)
        arr_extra = self._arrival_extra(w, i)

        self.since_refresh += 1
        if self.since_refresh >= cp.index_refresh:
            self._rebuild_index(cluster)
            self.since_refresh = 0
        else:
            self.desc = cluster.descriptors()

        cont_t = torch.as_tensor(cont)[None]
        type_t = torch.as_tensor([t])
        zt = self.net.embed_tasks(cont_t, type_t)[0].numpy()

        # --- step 2: candidates
        if self.v.btree_index:
            cand = self.retriever.query(zt)
            m = cp.avail_candidates
            if m:
                least = [j for _, j in zip(range(m), (v for _, v in self.avail_tree.items()))]
                cand = np.unique(np.r_[least, cand])
        else:
            cand = self.rng.choice(cluster.n, self.K - (K_EXPLORE if self.train else 0), replace=False)
        if self.train:
            extra = self.rng.choice(cluster.n, K_EXPLORE, replace=False)
            cand = np.unique(np.r_[cand, extra])[: self.K]
        cand = cand[cluster.cores[cand] >= w.cores[i]]
        if len(cand) == 0:
            cand = np.flatnonzero(cluster.cores >= w.cores[i])[: self.K]

        # --- step 3: weighted tradeoff filtering
        pred = cluster.predict(w, i, cand)
        omega, pair, lat = self._tradeoff(pred, w, i)
        ok = pred["feasible"]
        if self.v.multi_objective:
            sel = self._feasible(omega, lat, w.deadline[i], ok)
        else:
            sel = np.flatnonzero(ok)
        sel = sel[: self.K]
        nodes = cand[sel]

        # --- step 4: PPO decision over the feasible set
        k = len(nodes)
        res = np.zeros((self.K, 5), np.float32)
        prs = np.zeros((self.K, 9), np.float32)
        mask = np.zeros(self.K, bool)
        res[:k], prs[:k], mask[:k] = self.desc[nodes], pair[sel], True
        glob = np.array([self.desc[:, 0].mean(), arr_extra[1], arr_extra[0] / 3], np.float32)
        with torch.no_grad():
            logits, value = self.net(cont_t, type_t, torch.as_tensor(res)[None], torch.as_tensor(prs)[None],
                                     torch.as_tensor(mask)[None], torch.as_tensor(glob)[None])
        if self.train and self.pending is not None and self.learner.ready():
            self._finish_pending(done=False)
            self._learn(float(value[0]))
        elif self.pending is not None:
            self._finish_pending(done=False)
        probs = torch.softmax(logits[0], -1)
        a = int(torch.multinomial(probs, 1)) if self.train else int(torch.argmax(probs))
        j = int(nodes[a])

        # --- step 5: predictive prewarming on other top-ranked candidates
        pre_e0 = cluster.prewarm_energy_j
        lam, prewarmed = None, []
        if self.v.prewarming and k > 1:
            with torch.no_grad():
                lam = float(self.arr_head(torch.as_tensor(zt)[None], torch.as_tensor(arr_extra)[None]))
            order = torch.argsort(probs[:k], descending=True).numpy()
            picked = 0
            for o in order:
                if picked >= cp.prewarm_top_k:
                    break
                n_ = int(nodes[o])
                if n_ == j:
                    continue
                picked += 1
                cold_norm = cluster.cold[n_] / self.cfg.cluster.cold_start_s[1]   # Cold(R_j), Eq. (17)
                if lam * cold_norm >= cp.prewarm_threshold:                          # Eq. (18)
                    if cluster.prewarm(n_, int(w.ttype[i])):
                        self.prewarm_events += 1
                        prewarmed.append(n_)

        # decision trace for the dashboard's inspector (cheap: references to existing arrays)
        self.debug = dict(cand=cand, pred=pred, omega=omega, sel=sel, nodes=nodes, probs=probs[:k].numpy(),
                          chosen=j, lam=lam, prewarmed=prewarmed, tau=getattr(self.retriever, "tau", None),
                          value=float(value[0]))

        self.pending = dict(cont=cont, ttype=t, res=res, pair=prs, mask=mask, glob=glob, action=a,
                            logp=float(torch.log(probs[a] + 1e-12)), value=float(value[0]),
                            arr_extra=arr_extra, i=i, prewarm_j=cluster.prewarm_energy_j - pre_e0,
                            sustain=float(w.sustain[i]))
        self.decision_times.append(time.perf_counter() - t0)
        return j

    def observe(self, cluster: Cluster, w: Workload, i: int, rec: TaskRecord) -> None:
        p = self.pending
        eta = self.cfg.caprf.eta
        L = self.rmm["L"](rec.latency)
        if self.v.multi_objective:  # Eq. (13); carbon weighted by the task's sustainability preference
            E = self.rmm["E"](rec.energy_j + p["prewarm_j"])
            C = self.rmm["C"](rec.carbon_g)
            O = self.rmm["O"](rec.cold_s)
            r = eta[0] * rec.sla - eta[1] * L - eta[2] * E - eta[3] * (0.5 + p["sustain"]) * C - eta[4] * O
        else:
            r = eta[0] * rec.sla - eta[1] * L
        p["reward"] = r
        self.ep_reward += r
        # resource state changed -> update its keys in the B+-trees (O(log N))
        if self.v.btree_index:
            j = rec.node
            self.avail_tree.delete(self.avail_key[j])
            self.avail_key[j] = (float(cluster.core_free[j, : cluster.cores[j]].mean()), j)
            self.avail_tree.insert(self.avail_key[j], j)
            d = cluster.descriptors()[rec.node]
            y = self.net.embed_resources(torch.as_tensor(d)[None])[0].numpy()
            self.index.update(rec.node, y)

    def _finish_pending(self, done: bool) -> None:
        p, self.pending = self.pending, None
        if not self.train:
            return
        self.learner.rollout.add(cont=p["cont"], ttype=p["ttype"], res=p["res"], pair=p["pair"],
                                 mask=p["mask"], glob=p["glob"], action=p["action"], logp=p["logp"],
                                 value=p["value"], reward=p["reward"], done=done,
                                 arr_extra=p["arr_extra"], arr_label=self.arr_labels[p["i"]])
        if done and self.learner.ready():
            self._learn(0.0)

    def _learn(self, last_value: float) -> None:
        self.net.train()
        self.learner.update(last_value)
        self.net.train(self.train)
        self.retriever.tau = None
        self.res_emb = None  # encoder changed: re-key every resource on the next decision
        self.since_refresh = self.cfg.caprf.index_refresh

    # ------------------------------------------------------------------ persistence
    def eval_copy(self) -> "CAPRFScheduler":
        c = copy.copy(self)
        c.norm = copy.deepcopy(self.norm)
        c.scales = copy.deepcopy(self.scales)
        c.rmm = copy.deepcopy(self.rmm)
        c.rng = np.random.default_rng(self.cfg.seed + 5)
        c.episode_log, c.decision_times = [], []
        return c

    def snapshot(self) -> dict:
        return copy.deepcopy(dict(net=self.net.state_dict(), arr=self.arr_head.state_dict(), norm=self.norm,
                                  scales=self.scales, rmm=self.rmm))

    def restore(self, s: dict) -> None:
        self.net.load_state_dict(s["net"])
        self.arr_head.load_state_dict(s["arr"])
        self.norm, self.scales, self.rmm = copy.deepcopy(s["norm"]), copy.deepcopy(s["scales"]), copy.deepcopy(s["rmm"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(net=self.net.state_dict(), arr=self.arr_head.state_dict(), norm=self.norm.state(),
                        scales={k: v.v for k, v in self.scales.items()}, type_map=self.type_map,
                        variant=self.v.__dict__, log=self.learner.log), path)

    def load(self, path: Path) -> None:
        s = torch.load(path, weights_only=False)
        self.net.load_state_dict(s["net"])
        self.arr_head.load_state_dict(s["arr"])
        self.norm.load(s["norm"])
        for k, v in s["scales"].items():
            self.scales[k].v = v
        self.learner.log = s.get("log", [])
