"""Neural components of CAPRF++.

TaskEncoder      Eqs. (1)-(6): z-scored continuous features + learned type embedding -> MLP -> z_t
ResourceEncoder  Eq. (7): psi_w(a_j, e_j, l_j, gamma_j, is_cloud) -> z_r in the same latent space
PolicyValueNet   actor scores each feasible candidate; the logit includes -beta*||z_t - z_r||, so
                 PPO gradients pull rewarded task/resource pairs together (the implicit,
                 reward-driven metric learning of Sec. 3.2). Critic sees the pooled state.
ArrivalHead      Eq. (16): lambda_i = sigmoid(W x + b), probability another task of this type
                 arrives within a short window.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_TASK_CONT = 5   # deadline, compute, priority, sustainability, cores (normalised)
N_RES = 5         # availability, energy efficiency, latency, carbon, is_cloud
N_PAIR = 9        # L, E, C, O, Omega, warm, wait, latency/deadline, is_cloud
N_GLOBAL = 3      # cluster mean availability, arrival intensity, type arrival rate


class RunningNorm:
    """Eq. (1) z-score with moments over a moving window (exponentially weighted)."""

    def __init__(self, dim: int, window: int = 2000):
        self.a = 1.0 / window
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.n = 0

    def __call__(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.n += 1
            a = max(self.a, 1.0 / self.n)
            d = x - self.mean
            self.mean = self.mean + a * d
            self.var = (1 - a) * (self.var + a * d * d)
        return (x - self.mean) / np.sqrt(self.var + 1e-6)

    def state(self):
        return dict(mean=self.mean.tolist(), var=self.var.tolist(), n=self.n)

    def load(self, s):
        self.mean, self.var, self.n = np.array(s["mean"]), np.array(s["var"]), s["n"]


class TaskEncoder(nn.Module):
    def __init__(self, n_types: int, d: int, type_dim: int, hidden: int, context: bool = True):
        super().__init__()
        self.context = context
        self.n_types = n_types
        if context:  # learned embedding; index n_types is the UNK type used for unseen classes
            self.embed = nn.Embedding(n_types + 1, type_dim)
            self.net = nn.Sequential(nn.Linear(N_TASK_CONT + type_dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, d))
        else:  # ablation baseline: raw features + one-hot type, single linear map
            self.net = nn.Linear(N_TASK_CONT + n_types, d)

    def forward(self, cont: torch.Tensor, ttype: torch.Tensor) -> torch.Tensor:
        if self.context:
            return self.net(torch.cat([cont, self.embed(ttype)], -1))
        onehot = F.one_hot(ttype.clamp(max=self.n_types), self.n_types + 1)[..., : self.n_types].float()
        return self.net(torch.cat([cont, onehot], -1))


class ResourceEncoder(nn.Module):
    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_RES, hidden), nn.SiLU(), nn.Linear(hidden, d))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class PolicyValueNet(nn.Module):
    def __init__(self, n_types: int, d: int = 32, type_dim: int = 8, hidden: int = 256,
                 context: bool = True):
        super().__init__()
        self.task_enc = TaskEncoder(n_types, d, type_dim, 128, context)
        self.res_enc = ResourceEncoder(d)
        inp = 3 * d + N_PAIR
        self.actor = nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                   nn.Linear(hidden, 1))
        self.dist_scale = nn.Parameter(torch.tensor(0.0))
        self.critic = nn.Sequential(nn.Linear(d + 2 * hidden // 4 + N_GLOBAL, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.cand_proj = nn.Linear(d + N_PAIR, hidden // 4)

    def actor_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith(("critic", "cand_proj"))]

    def critic_params(self):
        return [p for n, p in self.named_parameters() if n.startswith(("critic", "cand_proj"))]

    def forward(self, cont, ttype, res, pair, mask, glob):
        """cont (B,5) ttype (B,) res (B,K,5) pair (B,K,9) mask (B,K) bool glob (B,3)
        -> logits (B,K) with -inf on padding, value (B,)."""
        zt = self.task_enc(cont, ttype)                       # (B, d)
        zr = self.res_enc(res)                                # (B, K, d)
        zt_k = zt.unsqueeze(1).expand_as(zr)
        h = torch.cat([zt_k, zr, zt_k * zr, pair], -1)
        dist = torch.linalg.vector_norm(zt_k - zr, dim=-1)
        logits = self.actor(h).squeeze(-1) - F.softplus(self.dist_scale) * dist
        logits = logits.masked_fill(~mask, -1e9)
        c = self.cand_proj(torch.cat([zr, pair], -1))         # (B, K, h/4)
        m = mask.unsqueeze(-1).float()
        mean = (c * m).sum(1) / m.sum(1).clamp(min=1)
        mx = c.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        value = self.critic(torch.cat([zt.detach(), mean, mx, glob], -1)).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def embed_tasks(self, cont, ttype):
        return self.task_enc(cont, ttype)

    @torch.no_grad()
    def embed_resources(self, res):
        return self.res_enc(res)


class ArrivalHead(nn.Module):
    """Eq. (16) on [z_t (detached), recent per-type arrival rate, global intensity]."""

    def __init__(self, d: int = 32):
        super().__init__()
        self.lin = nn.Linear(d + 2, 1)

    def forward(self, zt, extra):
        return torch.sigmoid(self.lin(torch.cat([zt, extra], -1))).squeeze(-1)
