"""
Small mapping test on MNIST label subsets (no FL, minutes on a laptop).

Parties get overlapping digit subsets with their OWN local label ids, e.g.
    P0 = {0,1,2,3,4} -> local 0..4
    P1 = {5,6,7,8}   -> local 0..3
    P2 = {0,1,5,6,9} -> local 0..4
Ground truth = the TRUE digit (never the displayed class name).

Per method and seed:
  mapping    precision / recall / F1 / FP over cross-party label pairs
  acc_strict global CNN trained on all parties' real data relabelled by the method's global ids;
             a test prediction is correct iff it is exactly the party's gid for the true label AND that
             gid is pure (all its members are the same digit).
               wrong merge  -> the impure gid scores 0          (over-merging costs)
               missed merge -> the same digit is split over two gids, the model can't tell which
                               one a party means -> ~half wrong (under-merging costs)
  acc_digit  lenient: correct iff the predicted gid's majority digit is the true digit
             (only wrong merges cost; kept for reference)

Settings:
  --seeds 5              repeat everything (data sampling, local models, PSI masks, global model) 5x -> mean / std
  --per_class 20         data-scarce: 20 training images per class per client (merging then really helps)
  --name_style mixed     parties alternate "0" and "zero" style names -> tests matching when names differ
  --desc_encoder st:all-MiniLM-L6-v2   semantic description encoder for psi_filter (pip install sentence-transformers);
                         default 'char' = character n-grams, which CANNOT match "0" with "zero"
  psi_attn / psi_attn_filter  descriptions -> --text_encoder (sentence-transformers, default all-MiniLM-L6-v2)
                         -> attention over public TEXT anchors (rt_protocol.TEXT_ANCHORS, no number words)
                         -> coordinates -> fuzzy PSI. attn_filter adds the same image filter as psi_filter.
  --descriptions short|long   class description text: the name itself, or a short sentence
                         ("the handwritten digit zero" / "the handwritten digit 0")

  python3 test_mnist_split_new.py --device mps --seeds 5
  python3 test_mnist_split_new.py --device mps --seeds 5 --per_class 20
  python3 test_mnist_split_new.py --device mps --seeds 5 --name_style mixed --desc_encoder st:all-MiniLM-L6-v2
Output: logs/mnist_split_test/<time>/per_seed.csv, summary.csv, results.png, mapping_<method>_seed<k>.txt
"""
import argparse
import csv
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

from data.datasets import get_raw_dataset_transform
from label_mapping.label_mapping_utils import (
    label_mapping, single_direction_label_mapping, feature_bi_direction_label_mapping,
    image_cosine_similarity_mapping, missing_link_label_mapping, get_real_images, clear_image_caches)
from label_mapping.rt_protocol import run_rt_protocol, to_group_map, pair_metrics, TEXT_ANCHORS
from utils.nets import get_heterogeneous_model

ALL_METHODS = ["psi_filter", "psi_affscan", "psi_attn", "psi_attn_filter", "image-bi", "image-single",
               "feature-bi", "image-cs", "missing_link", "class_name", "oracle", "independent"]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


class Log:
    def log(self, m):
        pass


class Relabel(Dataset):
    def __init__(self, base, idx, remap):
        self.base, self.idx, self.remap = base, list(idx), remap

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        x, y = self.base[self.idx[k]]
        return x, self.remap[int(y)]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="0,1,2,3,4;5,6,7,8;0,1,5,6,9")
    ap.add_argument("--clients_per_party", type=int, default=2)
    ap.add_argument("--samples_per_client", type=int, default=1000)
    ap.add_argument("--per_class", type=int, default=0, help="training images per class per client (overrides samples_per_client)")
    ap.add_argument("--archs", default="1,2", help="client model ids from utils.nets, cycled (1=CNN, 2=ResNet8). LeNet(6)/AlexNet(7) fail on mps")
    ap.add_argument("--local_epochs", type=int, default=None, help="default 3 (20 with --per_class)")
    ap.add_argument("--global_epochs", type=int, default=None, help="default 5 (30 with --per_class)")
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--name_style", default="digit", choices=["digit", "word", "mixed"])
    ap.add_argument("--desc_encoder", default="char", help="'char' or 'st:<sentence-transformers model>'")
    ap.add_argument("--ladder_desc", default=None, help="comma list; default depends on --desc_encoder")
    ap.add_argument("--text_encoder", default="all-MiniLM-L6-v2", help="sentence-transformers model for psi_attn*")
    ap.add_argument("--tau", type=float, default=0.05, help="attention temperature for psi_attn*")
    ap.add_argument("--ladder_attn", default=None, help="comma list for psi_attn* coordinates")
    ap.add_argument("--descriptions", default="short", choices=["short", "long"])
    ap.add_argument("--psi", default="he", choices=["he", "plain"])
    ap.add_argument("--encoder_weights", default="DEFAULT", help="'random' to skip the download")
    ap.add_argument("--entropy_ratio", type=float, default=0.25)
    ap.add_argument("--cs_threshold", type=float, default=0.9)
    ap.add_argument("--missing_threshold", type=float, default=0.25)
    ap.add_argument("--anchor_datasets", default="FashionMNIST,USPS,CIFAR100")
    ap.add_argument("--num_anchors", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--data_root", default="./data/raw")
    ap.add_argument("--out", default="logs/mnist_split_test")
    a = ap.parse_args()
    a.local_epochs = a.local_epochs or (20 if a.per_class else 3)      # few samples -> more epochs, same #steps scale
    a.global_epochs = a.global_epochs or (30 if a.per_class else 5)
    return a


def train(model, loader, epochs, device):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            _, logits = model(x.to(device))
            F.cross_entropy(logits, y.to(device)).backward()
            opt.step()
    return model.eval()


def party_names(k, digits, style):
    use_word = style == "word" or (style == "mixed" and k % 2 == 1)
    return [WORDS[d] if use_word else str(d) for d in digits]


def describe(name, how):
    return name if how == "short" else f"the handwritten digit {name}"


# ---------------------------------------------------------------- PSI inputs (same pipeline as server_new.py)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_CACHE = {}


def encoder(a, dev):
    if "enc" not in _CACHE:
        w = None if a.encoder_weights == "random" else a.encoder_weights
        enc = torchvision.models.resnet18(weights=w)
        enc.fc = nn.Identity()
        _CACHE["enc"] = enc.to(dev).eval()
    return _CACHE["enc"]


def st_model(name):
    key = "st:" + name
    if key not in _CACHE:
        from sentence_transformers import SentenceTransformer
        _CACHE[key] = SentenceTransformer(name, device="cpu")
    return _CACHE[key]


def text_vectors(a, texts):
    return st_model(a.text_encoder).encode(list(texts), normalize_embeddings=True)


def desc_vectors(a, names):
    if a.desc_encoder == "char":
        return None
    if "st" not in _CACHE:
        from sentence_transformers import SentenceTransformer
        _CACHE["st"] = SentenceTransformer(a.desc_encoder.split(":", 1)[1], device="cpu")
    return _CACHE["st"].encode([describe(n, a.descriptions) for n in names], normalize_embeddings=True)


def psi_inputs(a, parties, dev, seed, need_text):
    enc = encoder(a, dev)

    @torch.no_grad()
    def emb(x):
        x = F.interpolate(x.to(dev) * 0.5 + 0.5, size=64, mode="bilinear", align_corners=False)
        return enc((x - _MEAN.to(dev)) / _STD.to(dev)).cpu().numpy()

    rng = np.random.default_rng(seed)
    names = a.anchor_datasets.split(",")
    P = []
    for k, n in enumerate(names):
        ds = get_raw_dataset_transform(n, a.data_root, train=False)
        per = a.num_anchors // len(names) + (k < a.num_anchors % len(names))
        idx = rng.choice(len(ds), per, replace=False)
        P.append(emb(torch.stack([ds[int(i)][0] for i in idx])))
    P = np.concatenate(P)
    mu = P.mean(0)
    clients, owner = {}, {}
    for p in parties:
        dv = desc_vectors(a, p["names"])
        for c in p["clients"]:
            summ = {l: emb(torch.stack([x for x, y in c["data"] if y == l][:64])).mean(0) - mu
                    for l in range(len(p["digits"]))}
            clients[c["cid"]] = {"summ": summ, "names": dict(enumerate(p["names"])),
                                 "count": {l: 10 ** 6 for l in summ}}
            if dv is not None:
                clients[c["cid"]]["desc_vecs"] = {l: dv[l] for l in summ}
            owner[c["cid"]] = p["name"]
    K_text = None
    if need_text:                                   # attn*: every client's enc_i = the public text encoder
        K_text = text_vectors(a, TEXT_ANCHORS)
        for p in parties:
            tv = text_vectors(a, [describe(n, a.descriptions) for n in p["names"]])
            for c in p["clients"]:
                clients[c["cid"]]["text_vecs"] = {l: tv[l] for l in range(len(p["digits"]))}
    return clients, owner, P - mu, K_text


# ---------------------------------------------------------------- one seed
def run_seed(a, seed, dev, train_ds, test_ds, out):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ytr, yte = np.asarray(train_ds.targets), np.asarray(test_ds.targets)
    rng = np.random.default_rng(seed)
    pool = {d: list(rng.permutation(np.flatnonzero(ytr == d))) for d in range(10)}
    archs = [int(v) for v in a.archs.split(",")]

    parties, cid = [], 0
    for k, spec in enumerate(a.splits.split(";")):
        digits = sorted(int(v) for v in spec.split(","))
        remap = {d: l for l, d in enumerate(digits)}
        p = {"name": f"P{k}", "digits": digits, "names": party_names(k, digits, a.name_style), "clients": []}
        per = a.per_class or a.samples_per_client // len(digits)
        for _ in range(a.clients_per_party):
            idx = [pool[d].pop() for d in digits for _ in range(per)]
            data = Relabel(train_ds, idx, remap)
            model = get_heterogeneous_model(archs[cid % len(archs)], 3, len(digits), 32, 256)
            train(model, DataLoader(data, 64, shuffle=True), a.local_epochs, dev)
            p["clients"].append({"cid": cid, "data": [data[i] for i in range(len(data))], "model": model})
            cid += 1
        p["test"] = DataLoader(Relabel(test_ds, np.flatnonzero(np.isin(yte, digits)), remap), 256)
        parties.append(p)

    ids = [p["name"] for p in parties]
    lsm = {p["name"]: p["names"] for p in parties}                       # what methods may see
    truth = {p["name"]: p["digits"] for p in parties}                     # scoring only
    clients_dict = {p["name"]: [c["model"].to(dev) for c in p["clients"]] for p in parties}
    img_kw = dict(args=a, test_loaders={p["name"]: p["test"] for p in parties})
    logger, psi = Log(), {}

    def mapping_of(method):
        clear_image_caches()
        if method.startswith("psi_"):
            if not psi:
                psi["v"] = psi_inputs(a, parties, dev, seed,
                                      need_text=any(m.startswith("psi_attn") for m in a.methods.split(",")))
            clients, owner, P, K_text = psi["v"]
            meth = method[4:]
            if meth.startswith("attn"):                 # attn* reads descriptions through the text encoder
                clients = {i: {**c, "desc_vecs": c["text_vecs"]} for i, c in clients.items()}
            ld = ([float(v) for v in a.ladder_desc.split(",")] if a.ladder_desc else
                  [0.95, 0.9, 0.85, 0.8] if a.desc_encoder == "char" else
                  np.round(np.arange(0.9, 0.45 - 1e-9, -0.05), 2).tolist())
            lm = ([float(v) for v in a.ladder_attn.split(",")] if a.ladder_attn else
                  np.round(np.arange(0.90, 0.40 - 1e-9, -0.05), 2).tolist())
            ladder = {"desc": ld, "aff": [0.99, 0.985, 0.98, 0.975, 0.97, 0.965, 0.96, 0.955, 0.95], "main": lm}
            table, _, _ = run_rt_protocol(clients, P, method=meth, psi=a.psi, ladder=ladder, tau=a.tau,
                                          min_samples=0, seed=seed, log=lambda *_: None, K_text=K_text)
            return to_group_map(table, owner)
        if method == "image-bi":
            return label_mapping(get_real_images, ids, clients_dict, lsm, a.entropy_ratio, True, logger, **img_kw)
        if method == "image-single":
            return single_direction_label_mapping(get_real_images, ids, clients_dict, lsm, a.entropy_ratio, True,
                                                  logger, **img_kw)
        if method == "feature-bi":
            feats = {}
            with torch.no_grad():
                for p in parties:
                    feats[p["name"]] = {l: torch.cat([c["model"](torch.stack(
                        [x for x, y in c["data"] if y == l]).to(dev))[0].mean(0, keepdim=True)
                        for c in p["clients"]]) for l in range(len(p["digits"]))}
            return feature_bi_direction_label_mapping(feats, ids, clients_dict, lsm, a.entropy_ratio, True, logger)
        if method == "image-cs":
            return image_cosine_similarity_mapping(get_real_images, ids, lsm, a.cs_threshold, logger, **img_kw)
        if method == "missing_link":
            return missing_link_label_mapping(get_real_images, ids, clients_dict, lsm, a.missing_threshold,
                                              logger, **img_kw)
        if method in ("class_name", "oracle"):   # class_name: exact displayed names; oracle: true digits
            src = lsm if method == "class_name" else {d: [str(v) for v in truth[d]] for d in ids}
            names = sorted({n for v in src.values() for n in v})
            return {d: {l: names.index(n) for l, n in enumerate(src[d])} for d in ids}
        if method == "independent":
            g = iter(range(1000))
            return {d: {l: next(g) for l in range(len(lsm[d]))} for d in ids}
        raise ValueError(method)

    def global_acc(mapping):
        # canonical gid order (by first member (party, local id)): identical mappings -> identical
        # training labels -> identical accuracy, whatever numbering a method happened to emit
        first = {}
        for d in ids:
            for l, g in sorted(mapping.get(d, {}).items()):
                first.setdefault(g, (ids.index(d), l))
        gids = sorted(first, key=first.get)
        dense = {g: k for k, g in enumerate(gids)}
        members = defaultdict(set)
        for d, mp in mapping.items():
            for l, g in mp.items():
                members[dense[g]].add(truth[d][l])
        pure = {g: len(s) == 1 for g, s in members.items()}
        xs, ys, votes = [], [], defaultdict(lambda: np.zeros(10))
        for p in parties:
            mp = mapping.get(p["name"], {})
            for c in p["clients"]:
                for x, y in c["data"]:
                    if y in mp:
                        xs.append(x)
                        ys.append(dense[mp[y]])
                        votes[dense[mp[y]]][p["digits"][y]] += 1
        gid_digit = {g: int(v.argmax()) for g, v in votes.items()}
        torch.manual_seed(seed)
        model = get_heterogeneous_model(1, 3, len(gids), 32, 256)
        train(model, DataLoader(list(zip(xs, ys)), 64, shuffle=True), a.global_epochs, dev)
        strict, lenient = {}, {}
        with torch.no_grad():
            for p in parties:
                mp = mapping.get(p["name"], {})
                s = l_ok = n = 0
                for x, y in p["test"]:
                    pred = model(x.to(dev))[1].argmax(1).cpu()
                    for pr, t in zip(pred.tolist(), y.tolist()):
                        n += 1
                        g = dense.get(mp.get(t))
                        s += int(g is not None and pr == g and pure[g])
                        l_ok += int(gid_digit.get(pr) == p["digits"][t])
                strict[p["name"]], lenient[p["name"]] = 100.0 * s / n, 100.0 * l_ok / n
        return float(np.mean(list(strict.values()))), float(np.mean(list(lenient.values()))), len(gids)

    rows = []
    for method in a.methods.split(","):
        t = time.time()
        try:
            mapping = mapping_of(method)
        except Exception as e:
            print(f"  [{method}] FAILED: {type(e).__name__}: {e}")
            continue
        assign = {(d, l): g for d, m in mapping.items() for l, g in m.items()}
        m = pair_metrics(assign, lambda k1, k2: truth[k1[0]][k1[1]] == truth[k2[0]][k2[1]])
        acc_s, acc_d, n_g = global_acc(mapping)
        rows.append({"seed": seed, "method": method, "f1": m["f1_score"], "precision": m["precision"],
                     "recall": m["recall"], "FP": m["FP"], "FN": m["FN"], "acc_strict": acc_s,
                     "acc_digit": acc_d, "n_gids": n_g, "seconds": round(time.time() - t, 1)})
        with open(os.path.join(out, f"mapping_{method}_seed{seed}.txt"), "w") as f:
            for g in sorted({g for mp in mapping.values() for g in mp.values()}):
                f.write(f"gid {g}: " + ", ".join(f"{d}:'{lsm[d][l]}'(digit {truth[d][l]})" for d, mp in mapping.items()
                                                 for l, gg in mp.items() if gg == g) + "\n")
        print(f"  [{method:13s}] F1={m['f1_score']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} FP={m['FP']} "
              f"| strict {acc_s:6.2f}% digit {acc_d:6.2f}% | {n_g} gids | {time.time() - t:.0f}s")
    return rows


# ---------------------------------------------------------------- main
def main():
    a = parse()
    dev = a.device
    out = os.path.join(a.out, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out, exist_ok=True)
    train_ds = get_raw_dataset_transform("MNIST", a.data_root, train=True)
    test_ds = get_raw_dataset_transform("MNIST", a.data_root, train=False)
    with open(os.path.join(out, "config.txt"), "w") as f:
        f.write("\n".join(f"{k}: {v}" for k, v in vars(a).items()) + "\n")

    rows = []
    for s in range(a.seed, a.seed + a.seeds):
        print(f"=== seed {s} ===")
        rows += run_seed(a, s, dev, train_ds, test_ds, out)

    with open(os.path.join(out, "per_seed.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    keys = ["f1", "precision", "recall", "FP", "acc_strict", "acc_digit"]
    summ = []
    for method in dict.fromkeys(r["method"] for r in rows):
        rs = [r for r in rows if r["method"] == method]
        d = {"method": method, "n_seeds": len(rs)}
        for k in keys:
            v = np.array([r[k] for r in rs], float)
            d[f"{k}_mean"], d[f"{k}_std"] = v.mean(), v.std()
        summ.append(d)
    summ.sort(key=lambda d: -d["acc_strict_mean"])
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)

    print(f"\nsplits={a.splits}  per_class={a.per_class or '-'}  names={a.name_style}  desc={a.desc_encoder}  "
          f"seeds={a.seeds}")
    print(f"{'method':<14}{'F1':>14}{'FP':>8}{'acc_strict %':>18}{'acc_digit %':>18}")
    for d in summ:
        print(f"{d['method']:<14}{d['f1_mean']:>7.3f}±{d['f1_std']:<6.3f}{d['FP_mean']:>6.1f}"
              f"{d['acc_strict_mean']:>11.2f}±{d['acc_strict_std']:<6.2f}{d['acc_digit_mean']:>11.2f}±{d['acc_digit_std']:<6.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [d["method"] for d in summ]
    col = ["#2a78d6" if n.startswith("psi_") else "#8a8984" for n in names]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for x, (k, title, lim) in zip(ax, [("f1", "Mapping F1", 1.05), ("acc_strict", "Global acc, strict (%)", 105),
                                       ("acc_digit", "Global acc, digit (%)", 105)]):
        x.barh(names, [d[f"{k}_mean"] for d in summ], xerr=[d[f"{k}_std"] for d in summ], color=col,
               error_kw={"elinewidth": 1, "capsize": 3})
        x.set_xlim(0, lim)
        x.set_title(title, loc="left", fontsize=11)
        x.spines[["top", "right"]].set_visible(False)
        x.grid(axis="x", alpha=0.3)
    ax[0].invert_yaxis()
    fig.suptitle(f"MNIST split {a.splits} | per_class={a.per_class or '-'} | names={a.name_style} | "
                 f"desc={a.desc_encoder} | mean ± std over {a.seeds} seed(s) | blue = PSI",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "results.png"), dpi=150)
    print(f"\nsaved: {out}/summary.csv, per_seed.csv, results.png")


if __name__ == "__main__":
    main()
