import numpy as np

from caprf.index.bplustree import BPlusTree, IDistanceIndex


def _check_invariants(t: BPlusTree):
    keys = [k for k, _ in t.items()]
    assert keys == sorted(keys) and len(keys) == t.size


def test_insert_delete_range_matches_sorted_list():
    rng = np.random.default_rng(0)
    for order in (4, 5, 16):
        t, ref = BPlusTree(order), {}
        for step in range(4000):
            k = float(rng.integers(0, 600))
            if rng.random() < 0.6:
                t.insert(k, step)
                ref[k] = step
            else:
                assert t.delete(k) == (k in ref)
                ref.pop(k, None)
            if step % 500 == 0:
                _check_invariants(t)
        _check_invariants(t)
        assert dict(t.items()) == ref
        lo, hi = 100.0, 300.0
        assert [k for k, _ in t.range(lo, hi)] == sorted(k for k in ref if lo <= k <= hi)


def test_bulk_load_then_mutate():
    pairs = [(float(i), i) for i in range(1000)]
    t = BPlusTree.bulk_load(pairs, order=8)
    assert list(t.items()) == pairs
    for i in range(0, 1000, 3):
        assert t.delete(float(i))
    t.insert(1.5, -1)
    _check_invariants(t)
    assert t.get(1.5) == -1 and t.get(3.0) is None


def test_idistance_equals_brute_force():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(500, 8))
    idx = IDistanceIndex(n_refs=4, order=8)
    idx.build(emb)
    for _ in range(30):
        j = int(rng.integers(500))
        emb[j] = rng.normal(size=8)
        idx.update(j, emb[j])
    for _ in range(50):
        q, tau = rng.normal(size=8), float(rng.uniform(1, 4))
        ids, d = idx.query(q, tau)
        brute = np.flatnonzero(np.linalg.norm(emb - q, axis=1) <= tau)
        assert set(ids.tolist()) == set(brute.tolist())
        assert np.all(np.diff(d) >= 0)
