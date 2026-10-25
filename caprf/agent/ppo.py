"""Learner side (Algorithm 2): clipped PPO with GAE, plus the FIFO trajectory buffer with
prioritised sampling (Sec. 3.4, Fig. 2).

Each update uses the newest on-policy rollout and, when the trajectory buffer is enabled,
an equally sized prioritised sample of recent transitions from the FIFO buffer (capacity N,
oldest evicted first). Replayed samples carry their behaviour log-prob mu(a|s), so the
PPO ratio pi/mu and its clipping keep the off-policy correction bounded; sampling
priorities are |advantage|^alpha with importance weights (PER, Schaul et al. 2016).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from caprf.config import PPOConfig
from caprf.models.networks import N_GLOBAL, N_PAIR, N_RES, N_TASK_CONT

FIELDS = {  # name -> (shape suffix, dtype)
    "cont": ((N_TASK_CONT,), np.float32), "ttype": ((), np.int64), "res": (("K", N_RES), np.float32),
    "pair": (("K", N_PAIR), np.float32), "mask": (("K",), bool), "glob": ((N_GLOBAL,), np.float32),
    "action": ((), np.int64), "logp": ((), np.float32), "value": ((), np.float32),
    "reward": ((), np.float32), "done": ((), bool), "arr_extra": ((2,), np.float32),
    "arr_label": ((), np.float32), "adv": ((), np.float32), "ret": ((), np.float32),
}


class Storage:
    """Fixed-capacity ring buffer of transitions (used for both the rollout and the FIFO buffer)."""

    def __init__(self, cap: int, K: int):
        self.cap, self.K = cap, K
        self.data = {k: np.zeros((cap,) + tuple(K if s == "K" else s for s in shp), dt)
                     for k, (shp, dt) in FIELDS.items()}
        self.ptr = 0
        self.full = False

    def __len__(self):
        return self.cap if self.full else self.ptr

    def add(self, **tr):
        i = self.ptr
        for k, v in tr.items():
            self.data[k][i] = v
        self.ptr = (self.ptr + 1) % self.cap
        self.full = self.full or self.ptr == 0

    def extend(self, other: "Storage", n: int):
        for s in range(n):
            self.add(**{k: other.data[k][s] for k in FIELDS})

    def clear(self):
        self.ptr, self.full = 0, False

    def batch(self, idx) -> dict:
        return {k: torch.as_tensor(v[idx]) for k, v in self.data.items()}


def gae(rew, val, done, last_value, gamma, lam):
    T = len(rew)
    adv = np.zeros(T, np.float32)
    nxt_v, run = last_value, 0.0
    for t in range(T - 1, -1, -1):
        nonterm = 1.0 - float(done[t])
        delta = rew[t] + gamma * nxt_v * nonterm - val[t]
        run = delta + gamma * lam * nonterm * run
        adv[t] = run
        nxt_v = val[t]
    return adv, adv + val


class PPOLearner:
    def __init__(self, net, arrival_head, cfg: PPOConfig, K: int, use_buffer: bool, seed: int):
        self.net, self.arr, self.cfg = net, arrival_head, cfg
        self.opt_pi = torch.optim.Adam(net.actor_params(), lr=cfg.lr_policy)
        self.opt_v = torch.optim.Adam(net.critic_params(), lr=cfg.lr_value)
        self.opt_arr = torch.optim.Adam(arrival_head.parameters(), lr=1e-3)
        self.rollout = Storage(cfg.rollout, K)
        self.buffer = Storage(cfg.buffer_size, K) if use_buffer else None
        self.rng = np.random.default_rng(seed)
        self.log: list[dict] = []

    def ready(self) -> bool:
        return len(self.rollout) >= self.cfg.rollout

    def update(self, last_value: float) -> dict:
        c, r = self.cfg, self.rollout
        n = len(r)
        adv, ret = gae(r.data["reward"][:n], r.data["value"][:n], r.data["done"][:n], last_value,
                       c.gamma, c.gae_lambda)
        r.data["adv"][:n], r.data["ret"][:n] = adv, ret

        # assemble the training set: on-policy rollout (+ prioritised replay from the FIFO buffer)
        parts = [(r, np.arange(n), np.ones(n, np.float32))]
        if self.buffer is not None and len(self.buffer) > c.minibatch:
            m = int(n * c.replay_frac / (1 - c.replay_frac))
            pr = (np.abs(self.buffer.data["adv"][: len(self.buffer)]) + 1e-3) ** c.per_alpha
            p = pr / pr.sum()
            idx = self.rng.choice(len(self.buffer), size=min(m, len(self.buffer)), p=p)
            w = (len(self.buffer) * p[idx]) ** -0.4
            parts.append((self.buffer, idx, (w / w.max()).astype(np.float32)))
        batches = [(src.batch(idx), torch.as_tensor(wt)) for src, idx, wt in parts]
        data = {k: torch.cat([b[k] for b, _ in batches]) for k in FIELDS}
        weight = torch.cat([wt for _, wt in batches])
        N = len(weight)

        stats = dict(pi_loss=0.0, v_loss=0.0, entropy=0.0, clipfrac=0.0, arr_loss=0.0, n=0)
        for _ in range(c.epochs):
            perm = torch.randperm(N)
            for s in range(0, N, c.minibatch):
                b = perm[s:s + c.minibatch]
                logits, value = self.net(data["cont"][b], data["ttype"][b], data["res"][b],
                                         data["pair"][b], data["mask"][b], data["glob"][b])
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(1, data["action"][b, None]).squeeze(1)
                probs = logp_all.exp()
                ent = -(probs * logp_all.masked_fill(~data["mask"][b], 0)).sum(-1)
                a = data["adv"][b]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = torch.exp(logp - data["logp"][b])
                surr = torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a)
                wb = weight[b]
                pi_loss = -(wb * surr).mean() - c.ent_coef * ent.mean()
                v_loss = (wb * (value - data["ret"][b]) ** 2).mean()

                self.opt_pi.zero_grad()
                self.opt_v.zero_grad()
                (pi_loss + c.vf_coef * v_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt_pi.step()
                self.opt_v.step()

                # arrival predictor (Eq. 16), supervised by observed future arrivals
                with torch.no_grad():
                    zt = self.net.task_enc(data["cont"][b], data["ttype"][b])
                lam = self.arr(zt, data["arr_extra"][b])
                arr_loss = F.binary_cross_entropy(lam.clamp(1e-5, 1 - 1e-5), data["arr_label"][b])
                self.opt_arr.zero_grad()
                arr_loss.backward()
                self.opt_arr.step()

                stats["pi_loss"] += float(pi_loss)
                stats["v_loss"] += float(v_loss)
                stats["entropy"] += float(ent.mean())
                stats["clipfrac"] += float(((ratio - 1).abs() > c.clip).float().mean())
                stats["arr_loss"] += float(arr_loss)
                stats["n"] += 1

        if self.buffer is not None:
            self.buffer.extend(r, n)
        r.clear()
        k = max(stats.pop("n"), 1)
        out = {key: v / k for key, v in stats.items()}
        out["samples"] = N
        self.log.append(out)
        return out
