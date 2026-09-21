"""circuit-PSI (cpsi_helper / cpsi_2pc / cpsi_tag) == plaintext precision protocol (same partition), plus MPC/HE primitive checks.

    python3 tests/test_circuit_psi_new.py          # ~30 s, needs tenseal
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from label_mapping import rt_protocol as rp  # noqa: E402
from label_mapping.circuit_psi_new import MPC, QSCALE, dec, enc, psi_shares, run_rt_protocol  # noqa: E402


def make(seed, n_clients=4, n_concepts=8, per_client=5, D=48, A=40, noise=0.15, img_noise=0.08):
    """Clients share concepts; each client has its OWN text encoder (random rotation)."""
    g = np.random.default_rng(seed)
    concept_img, concept_txt = g.normal(size=(n_concepts, D)), g.normal(size=(n_concepts, 24))
    anchors, P = g.normal(size=(A, 24)), g.normal(size=(60, D))
    clients = {}
    for i in range(n_clients):
        rot = np.linalg.qr(g.normal(size=(24, 24)))[0]
        own = g.choice(n_concepts, per_client, replace=False)
        c = clients[i] = {"summ": {}, "count": {}, "keyword_vecs": {}, "keywords": {}, "names": {},
                          "anchor_vecs": anchors @ rot}
        for a, k in enumerate(own):
            c["summ"][a] = concept_img[k] + img_noise * g.normal(size=D)
            c["keyword_vecs"][a] = np.stack([anchors[0] + .1 * g.normal(size=24),
                                             concept_txt[k] + noise * g.normal(size=24)]) @ rot
            c["keywords"][a], c["count"][a], c["names"][a] = ["dom", f"c{k}"], 50, str(k)
    return clients, P - P.mean(0)


def partition(t):
    return sorted(sorted(k for k in t if t[k] == g) for g in set(t.values()))


def main():
    m = MPC()
    x = np.array([-5, -1, 0, 1, 7, -2 ** 40, 2 ** 40, -2 ** 62, 2 ** 62])
    assert (dec(m.open(m.ltz(m.share(enc(x))))) == (x < 0)).all()
    rng = np.random.default_rng(1)
    qi, qj = rp.unit(rng.normal(size=(4, 30))), rp.unit(rng.normal(size=(3, 30)))
    qk = rp.unit(rng.normal(size=(4, 7))), rp.unit(rng.normal(size=(3, 7)))
    out = psi_shares({"x": {0: qi, 1: qj}, "y": {0: qk[0], 1: qk[1]}}, [(0, 1)], m, rng, defaultdict(int))
    Qk = [np.rint(q * QSCALE).astype(np.int64) for q in qk]
    assert (dec(m.open(out["y"][(0, 1)])) == Qk[0] @ Qk[1].T).all()   # signals stay separate
    C = out["x"][(0, 1)]
    Qi, Qj = np.rint(qi * QSCALE).astype(np.int64), np.rint(qj * QSCALE).astype(np.int64)
    assert (dec(m.open(C)) == Qi @ Qj.T).all()

    ladder = {"main": rp.LADDER.tolist(), "aff": [.99, .985, .98, .975, .97, .965, .96, .955, .95]}
    kw = dict(method="attn_filter", attn_match="precision", ladder=ladder, min_samples=0, log=lambda *_: None)
    for seed, hard in ((0, False), (1, False), (2, True)):
        extra = dict(n_clients=5, n_concepts=10, per_client=6, noise=.3, img_noise=.12) if hard else {}
        clients, P = make(seed, **extra)
        plain, edges, _ = run_rt_protocol(clients, P, psi="plain", seed=seed, **kw)
        for psi in ("cpsi_helper", "cpsi_2pc", "cpsi_tag"):
            circ, _, _ = run_rt_protocol(clients, P, psi=psi, seed=seed, **kw)
            assert partition(plain) == partition(circ), f"seed {seed}: {psi} table != plaintext table"
        print(f"seed {seed}: {len(edges)} edges, {len(set(plain.values()))} global ids -> identical")
    print("ok")


if __name__ == "__main__":
    main()
