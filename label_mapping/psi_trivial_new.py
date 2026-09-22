"""
psi_trivial_new.py -- rt_method: trivial  ("PSI-Trivial", run tag rt_psi_trivial).

Sanity / upper-bound baseline for the relation table: every client describes each of its classes by the
SAME agreed name ("3" for 3, "A" for A, "car" for car), so an EXACT PSI on the names gives the table directly.
No encoders, no public anchors, no fuzzy matching.

  name      = shared_name(class_name, aliases): strip, alias ("automobile" -> "car" via rt_gt_aliases),
              lower-case except single characters (EMNIST 'a' != 'A'); same rule as the ground truth.
  psi="dh"  (default) exact DH-PSI per client pair over RFC 3526 group 14 (2048-bit MODP, QR subgroup):
              client i sends {H(x)^a_i}, client j returns {H(x)^(a_i b_j)} and {H(y)^b_j} (shuffled),
              i computes {H(y)^(b_j a_i)} and intersects. a_i is one key per client (256-bit short exponent),
              so {H(x)^a_i} is computed once and reused for every pair.
  psi="plain" plain set intersection on the names (debug, same result).

Each exact match between client i and client j becomes an edge; global_table() unions them into global ids.
Every class a client holds takes part (min_samples=1): the name is known whatever the sample count, so
the rt_min_samples cut of the fuzzy methods (which need enough images for a stable summary) does not apply.
ponytail: all parties simulated in one process; in deployment i reports its matched (own id, j's masked id)
pairs to the server, j sends the permutation-free list only to i.
"""
import hashlib
import secrets
import time
from collections import Counter

import numpy as np

from label_mapping.rt_protocol import global_table

# RFC 3526, 2048-bit MODP group 14: safe prime p = 2q + 1
P_MODP = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
EXP_BITS = 256


def shared_name(s, aliases=None):
    s = str(s).strip()
    s = (aliases or {}).get(s, s)
    return s if len(s) == 1 else s.lower()


def hash_to_group(x):
    """H(x) in the prime-order subgroup of quadratic residues mod p (square of a 2048+ bit hash)."""
    h = int.from_bytes(hashlib.shake_256(b"rt_psi_trivial|" + x.encode("utf-8")).digest(264), "big") % P_MODP
    return pow(h, 2, P_MODP)


def dh_psi(sent_i, sent_j, key_i, key_j, rng):
    """One exact DH-PSI between i and j, given each side's own first message sent_k = [H(x)^a_k]
    (computed once per client, reused for every pair). j re-exponentiates i's list in order and
    sends its own list shuffled; i re-exponentiates j's list and intersects.
    Returns [(index into i's list, index into j's list)] of equal elements."""
    double_i = [pow(v, key_j, P_MODP) for v in sent_i]              # j -> i, same order
    perm = rng.permutation(len(sent_j))                             # j shuffles its own set
    double_j = {pow(sent_j[k], key_i, P_MODP): int(k) for k in perm}
    return [(a, double_j[v]) for a, v in enumerate(double_i) if v in double_j]


def run_trivial(clients, psi="dh", min_samples=1, seed=0, aliases=None, log=print, masked_out=None, **_):
    """clients {i: {"names": {a: class name}, "count": {a: n}}}  (other keys ignored).
    Returns (table {(i, a): gid}, edges [(i, a, j, b, 1)], diag) like rt_protocol.run_rt_protocol."""
    if psi not in ("dh", "plain"):
        raise ValueError("rt_method=trivial supports rt_psi: dh | plain")
    rng = np.random.default_rng(seed)
    t0 = time.time()
    elig, desc = {}, {}
    for i, c in clients.items():
        e = [a for a in sorted(c["names"]) if c["count"].get(a, 0) >= min_samples]
        elig[i] = [e[k] for k in rng.permutation(len(e))]             # masked id = position in this list
        desc[i] = [shared_name(c["names"][a], aliases) for a in elig[i]]
        dup = [n for n, k in Counter(desc[i]).items() if k > 1]
        if dup:
            raise ValueError(f"client {i}: several local classes share the description {dup}")

    ids = sorted(i for i in clients if elig[i])
    pairs = [(i, j) for x, i in enumerate(ids) for j in ids[x + 1:]]
    n_exp = 0
    if psi == "dh":
        keys = {i: secrets.randbits(EXP_BITS) | 1 for i in ids}
        sent = {i: [pow(hash_to_group(n), keys[i], P_MODP) for n in desc[i]] for i in ids}   # once per client
        n_exp = sum(len(v) for v in sent.values())
    edges = []
    for i, j in pairs:
        if psi == "dh":
            hits = dh_psi(sent[i], sent[j], keys[i], keys[j], rng)
            n_exp += len(desc[i]) + len(desc[j])
        else:
            pos = {n: b for b, n in enumerate(desc[j])}
            hits = [(a, pos[n]) for a, n in enumerate(desc[i]) if n in pos]
        edges += [(i, elig[i][a], j, elig[j][b], 1) for a, b in hits]

    all_ids = {i: sorted(set(c["count"]) | set(c["names"])) for i, c in clients.items()}
    table = global_table(edges, all_ids)
    if masked_out is not None:
        for i in clients:
            rest = [a for a in all_ids[i] if a not in set(elig[i])]
            masked_out.update({(i, a): m for m, a in enumerate(elig[i] + rest)})
    secs = time.time() - t0
    log(f"[RT] trivial ({psi}): {len(pairs)} client pairs, {len(edges)} exact matches, "
        f"{len(set(table.values()))} global ids, {n_exp} modexp, {secs:.1f}s")
    diag = {"candidates": np.array(len(edges)), "client_pairs": np.array(len(pairs)),
            "modexp": np.array(n_exp), "psi_seconds": np.array(int(round(secs)))}
    return table, edges, diag
