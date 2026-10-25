"""B+-tree and an iDistance metric index built on it (Sec. 3.2).

A B+-tree orders one-dimensional keys, but CAPRF++ needs "resources whose embedding lies
within distance tau of the task embedding" (Eq. 9). The standard way to answer that with a
B+-tree is iDistance (Jagadish et al., ACM TODS 2005):

  * pick m reference points; every resource embedding y belongs to its nearest reference o_p
  * its 1-D key is  p * C + ||y - o_p||   (C larger than any distance, so partitions don't mix)
  * by the triangle inequality, ||y - q|| <= tau implies | ||y - o_p|| - ||q - o_p|| | <= tau,
    so a query only range-scans [ ||q-o_p|| - tau, ||q-o_p|| + tau ] inside each partition
    whose ball it touches, then checks the exact distance for those few survivors.

Search is O(log N + candidates), and inserting/deleting one resource is O(log N), which
gives the real-time updates the paper asks for when a node's state changes.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np


class _Leaf:
    __slots__ = ("keys", "vals", "next")

    def __init__(self):
        self.keys: list = []
        self.vals: list = []
        self.next: _Leaf | None = None


class _Inner:
    __slots__ = ("keys", "kids")

    def __init__(self):
        self.keys: list = []
        self.kids: list = []


class BPlusTree:
    """B+-tree mapping unique sortable keys to values; leaves are linked for range scans."""

    def __init__(self, order: int = 16):
        assert order >= 4
        self.order = order
        self.root: _Leaf | _Inner = _Leaf()
        self.size = 0

    # ---------------------------------------------------------------- lookup
    def _leaf_for(self, key) -> _Leaf:
        node = self.root
        while isinstance(node, _Inner):
            node = node.kids[bisect.bisect_right(node.keys, key)]
        return node

    def get(self, key, default=None):
        leaf = self._leaf_for(key)
        i = bisect.bisect_left(leaf.keys, key)
        return leaf.vals[i] if i < len(leaf.keys) and leaf.keys[i] == key else default

    def range(self, lo, hi):
        """Yield (key, value) for lo <= key <= hi in key order."""
        leaf = self._leaf_for(lo)
        i = bisect.bisect_left(leaf.keys, lo)
        while leaf is not None:
            while i < len(leaf.keys):
                if leaf.keys[i] > hi:
                    return
                yield leaf.keys[i], leaf.vals[i]
                i += 1
            leaf, i = leaf.next, 0

    def items(self):
        node = self.root
        while isinstance(node, _Inner):
            node = node.kids[0]
        while node is not None:
            yield from zip(node.keys, node.vals)
            node = node.next

    def height(self) -> int:
        h, node = 1, self.root
        while isinstance(node, _Inner):
            node, h = node.kids[0], h + 1
        return h

    # ---------------------------------------------------------------- insert
    def insert(self, key, val) -> None:
        split = self._insert(self.root, key, val)
        if split is not None:
            sep, right = split
            root = _Inner()
            root.keys, root.kids = [sep], [self.root, right]
            self.root = root

    def _insert(self, node, key, val):
        if isinstance(node, _Leaf):
            i = bisect.bisect_left(node.keys, key)
            if i < len(node.keys) and node.keys[i] == key:
                node.vals[i] = val
                return None
            node.keys.insert(i, key)
            node.vals.insert(i, val)
            self.size += 1
            if len(node.keys) < self.order:
                return None
            mid = len(node.keys) // 2
            right = _Leaf()
            right.keys, right.vals = node.keys[mid:], node.vals[mid:]
            node.keys, node.vals = node.keys[:mid], node.vals[:mid]
            right.next, node.next = node.next, right
            return right.keys[0], right
        i = bisect.bisect_right(node.keys, key)
        split = self._insert(node.kids[i], key, val)
        if split is None:
            return None
        sep, right = split
        node.keys.insert(i, sep)
        node.kids.insert(i + 1, right)
        if len(node.kids) <= self.order:
            return None
        mid = len(node.keys) // 2
        new = _Inner()
        up = node.keys[mid]
        new.keys, new.kids = node.keys[mid + 1:], node.kids[mid + 1:]
        node.keys, node.kids = node.keys[:mid], node.kids[:mid + 1]
        return up, new

    # ---------------------------------------------------------------- delete
    def delete(self, key) -> bool:
        found = self._delete(self.root, key)
        if isinstance(self.root, _Inner) and len(self.root.kids) == 1:
            self.root = self.root.kids[0]
        return found

    def _min_fill(self) -> int:
        return (self.order + 1) // 2 - 1

    def _delete(self, node, key) -> bool:
        if isinstance(node, _Leaf):
            i = bisect.bisect_left(node.keys, key)
            if i >= len(node.keys) or node.keys[i] != key:
                return False
            del node.keys[i], node.vals[i]
            self.size -= 1
            return True
        i = bisect.bisect_right(node.keys, key)
        child = node.kids[i]
        found = self._delete(child, key)
        if found:
            n_child = len(child.keys) if isinstance(child, _Leaf) else len(child.kids)
            if n_child < self._min_fill():
                self._rebalance(node, i)
        return found

    def _rebalance(self, parent: _Inner, i: int) -> None:
        child = parent.kids[i]
        left = parent.kids[i - 1] if i > 0 else None
        right = parent.kids[i + 1] if i + 1 < len(parent.kids) else None
        if isinstance(child, _Leaf):
            if left is not None and len(left.keys) > self._min_fill():
                child.keys.insert(0, left.keys.pop())
                child.vals.insert(0, left.vals.pop())
                parent.keys[i - 1] = child.keys[0]
            elif right is not None and len(right.keys) > self._min_fill():
                child.keys.append(right.keys.pop(0))
                child.vals.append(right.vals.pop(0))
                parent.keys[i] = right.keys[0]
            elif left is not None:  # merge child into left
                left.keys += child.keys
                left.vals += child.vals
                left.next = child.next
                del parent.keys[i - 1], parent.kids[i]
            elif right is not None:  # merge right into child
                child.keys += right.keys
                child.vals += right.vals
                child.next = right.next
                del parent.keys[i], parent.kids[i + 1]
            return
        if left is not None and len(left.kids) > self._min_fill() + 1:
            child.keys.insert(0, parent.keys[i - 1])
            child.kids.insert(0, left.kids.pop())
            parent.keys[i - 1] = left.keys.pop()
        elif right is not None and len(right.kids) > self._min_fill() + 1:
            child.keys.append(parent.keys[i])
            child.kids.append(right.kids.pop(0))
            parent.keys[i] = right.keys.pop(0)
        elif left is not None:
            left.keys += [parent.keys[i - 1]] + child.keys
            left.kids += child.kids
            del parent.keys[i - 1], parent.kids[i]
        elif right is not None:
            child.keys += [parent.keys[i]] + right.keys
            child.kids += right.kids
            del parent.keys[i], parent.kids[i + 1]

    # ---------------------------------------------------------------- bulk load
    @classmethod
    def bulk_load(cls, pairs, order: int = 16) -> "BPlusTree":
        """Build from (key, value) pairs in O(N log N) (sort) + O(N)."""
        t = cls(order)
        pairs = sorted(pairs)
        if not pairs:
            return t
        fill = max(order // 2, 2)
        leaves = []
        for s in range(0, len(pairs), fill):
            lf = _Leaf()
            chunk = pairs[s:s + fill]
            lf.keys = [k for k, _ in chunk]
            lf.vals = [v for _, v in chunk]
            if leaves:
                leaves[-1].next = lf
            leaves.append(lf)
        if len(leaves) > 1 and len(leaves[-1].keys) < t._min_fill():  # keep the last leaf legal
            last = leaves.pop()
            leaves[-1].keys += last.keys
            leaves[-1].vals += last.vals
            leaves[-1].next = None
        level = leaves
        firsts = [lf.keys[0] for lf in level]
        while len(level) > 1:
            nxt, nfirst = [], []
            step = max(order // 2, 2)
            groups = [list(range(s, min(s + step, len(level)))) for s in range(0, len(level), step)]
            if len(groups) > 1 and len(groups[-1]) < 2:
                groups[-2] += groups.pop()
            for g in groups:
                n = _Inner()
                n.kids = [level[x] for x in g]
                n.keys = [firsts[x] for x in g[1:]]
                nxt.append(n)
                nfirst.append(firsts[g[0]])
            level, firsts = nxt, nfirst
        t.root = level[0]
        t.size = len(pairs)
        return t


# ============================================================================ iDistance
@dataclass
class QueryStats:
    scanned: int = 0      # entries returned by B+-tree range scans
    distance_evals: int = 0
    result: int = 0


class IDistanceIndex:
    """Metric (Euclidean) range search over resource embeddings using a B+-tree."""

    def __init__(self, n_refs: int = 4, order: int = 16, seed: int = 0):
        self.n_refs = n_refs
        self.order = order
        self.rng = np.random.default_rng(seed)
        self.tree = BPlusTree(order)
        self.emb: np.ndarray | None = None
        self.keys: dict[int, tuple] = {}

    def _assign(self, y: np.ndarray) -> tuple[int, float]:
        d = np.linalg.norm(self.refs - y, axis=1)
        p = int(np.argmin(d))
        return p, float(d[p])

    def build(self, emb: np.ndarray) -> None:
        """(Re)build from an (N, d) array; reference points chosen by farthest-point sampling."""
        self.emb = np.asarray(emb, dtype=np.float64).copy()
        n = len(self.emb)
        m = min(self.n_refs, n)
        refs = [int(self.rng.integers(n))]
        dmin = np.linalg.norm(self.emb - self.emb[refs[0]], axis=1)
        for _ in range(m - 1):
            refs.append(int(np.argmax(dmin)))
            dmin = np.minimum(dmin, np.linalg.norm(self.emb - self.emb[refs[-1]], axis=1))
        self.refs = self.emb[refs].copy()
        dall = np.linalg.norm(self.emb[:, None, :] - self.refs[None], axis=2)
        part = dall.argmin(1)
        dist = dall[np.arange(n), part]
        self.C = float(4 * dist.max() + 10.0)
        self.radius = np.zeros(m)
        np.maximum.at(self.radius, part, dist)
        self.keys = {j: (part[j] * self.C + dist[j], j) for j in range(n)}
        self.tree = BPlusTree.bulk_load([(k, j) for j, k in self.keys.items()], self.order)

    def update(self, j: int, y: np.ndarray) -> None:
        """Move one resource after its state (and therefore embedding) changed: O(log N)."""
        self.tree.delete(self.keys[j])
        self.emb[j] = y
        p, d = self._assign(y)
        if d > self.C / 4:  # embedding drifted far; keep partitions separable
            self.build(self.emb)
            return
        self.radius[p] = max(self.radius[p], d)
        self.keys[j] = (p * self.C + d, j)
        self.tree.insert(self.keys[j], j)

    def query(self, q: np.ndarray, tau: float, stats: QueryStats | None = None) -> tuple[np.ndarray, np.ndarray]:
        """All resources with ||y - q|| <= tau; returns (ids, distances) sorted by distance."""
        q = np.asarray(q, dtype=np.float64)
        dq = np.linalg.norm(self.refs - q, axis=1)
        ids = []
        for p in range(len(self.refs)):
            if dq[p] - tau > self.radius[p]:
                continue  # query ball misses this partition entirely
            lo = (p * self.C + max(dq[p] - tau, 0.0), -1)
            hi = (p * self.C + min(dq[p] + tau, self.radius[p]), 1 << 62)
            ids.extend(j for _, j in self.tree.range(lo, hi))
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            return ids, np.zeros(0)
        d = np.linalg.norm(self.emb[ids] - q, axis=1)
        keep = d <= tau
        if stats is not None:
            stats.scanned += len(ids)
            stats.distance_evals += len(ids) + len(self.refs)
            stats.result += int(keep.sum())
        order = np.argsort(d[keep])
        return ids[keep][order], d[keep][order]


class AdaptiveRetriever:
    """Eq. (9) with the adaptive threshold tau_i and the empty-set fallback of Sec. 3.2.

    tau tracks (EMA) the radius that returns about k_target resources; if a query returns
    fewer than k_min, tau is enlarged and the query retried (the paper's fallback); if it
    returns more than k_max, only the k_max nearest are kept.
    """

    def __init__(self, index: IDistanceIndex, k_min: int, k_max: int):
        self.index = index
        self.k_min, self.k_max = k_min, k_max
        self.k_target = (k_min + k_max) // 2
        self.tau: float | None = None
        self.stats = QueryStats()
        self.fallbacks = 0

    def query(self, q: np.ndarray) -> np.ndarray:
        if self.tau is None:  # initialise from the distance to the k_target-th neighbour
            d = np.sort(np.linalg.norm(self.index.emb - q, axis=1))
            self.tau = float(d[min(self.k_target, len(d) - 1)]) + 1e-6
        tau = self.tau
        for _ in range(12):
            ids, d = self.index.query(q, tau, self.stats)
            if len(ids) >= self.k_min or len(ids) == len(self.index.emb):
                break
            tau *= 1.5
            self.fallbacks += 1
        if len(ids) > self.k_target:
            r = float(d[self.k_target - 1])
        else:
            r = tau
        self.tau = 0.9 * self.tau + 0.1 * max(r, 1e-6)
        return ids[: self.k_max]
