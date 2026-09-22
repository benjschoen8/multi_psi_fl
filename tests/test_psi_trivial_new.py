"""rt_method=trivial: DH-PSI == plain intersection == ground truth (names shared across clients).

    python3 tests/test_psi_trivial_new.py        # a few seconds, numpy only
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from label_mapping.psi_trivial_new import run_trivial, shared_name  # noqa: E402


def partition(t):
    return sorted(sorted(k for k in t if t[k] == g) for g in set(t.values()))


def main():
    digits = [str(d) for d in range(10)]
    emnist = digits + [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["a", "b", "d"]
    cifar = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    spaces = [digits, emnist, cifar, ["Car", "Cat", "B", "b", "7"]]
    clients = {}
    for i in range(8):
        names = spaces[i % len(spaces)]
        clients[i] = {"names": dict(enumerate(names)),
                      "count": {a: (5 if (i == 1 and a == 3) else 50) for a in range(len(names))}}
    aliases = {"automobile": "car"}
    # default: every held class joins, however few samples -> exactly one gid per name
    t_all, _, _ = run_trivial(clients, psi="dh", aliases=aliases, seed=1)
    names_all = {shared_name(clients[i]["names"][a], aliases) for i, a in t_all}
    assert len(set(t_all.values())) == len(names_all), "one global id per distinct name"
    assert t_all[0, 3] == t_all[1, 3], "few-sample class must still match"

    t_dh, e_dh, _ = run_trivial(clients, psi="dh", aliases=aliases, seed=1, min_samples=10)
    t_pl, e_pl, _ = run_trivial(clients, psi="plain", aliases=aliases, seed=1, min_samples=10)
    assert partition(t_dh) == partition(t_pl), "dh != plain"
    assert sorted(e[:4] for e in e_dh) == sorted(e[:4] for e in e_pl)

    name = lambda i, a: shared_name(clients[i]["names"][a], aliases)
    for (i, a), g in t_dh.items():
        for (j, b), h in t_dh.items():
            if i == j:
                continue
            small = clients[i]["count"][a] < 10 or clients[j]["count"][b] < 10
            want = name(i, a) == name(j, b) and not small
            assert (g == h) == want, ((i, a), (j, b), g, h)
    assert t_dh[0, 3] != t_dh[1, 3], "below min_samples must keep its own gid"
    assert t_dh[2, 1] == t_dh[3, 0], "automobile == Car via alias + lower-case"
    assert t_dh[3, 2] != t_dh[3, 3], "'B' != 'b'"
    print(f"OK trivial: {len(set(t_dh.values()))} global ids, {len(e_dh)} edges, dh == plain == ground truth")


if __name__ == "__main__":
    main()
