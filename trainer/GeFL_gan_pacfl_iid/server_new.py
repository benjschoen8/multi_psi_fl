"""
Server for GeFL_gan_pacfl_iid + RT protocol (replacement for server.py; server.py untouched).

  * BEFORE round 1: RT protocol builds the relation table ONCE
    (label_mapping/rt_protocol.py; default method = Alg.2 description PSI + image-set affinity filter,
    real LabeledFuzzyPSI over CKKS).
  * DURING training: aggregate() only FedAvg's generator/discriminator per group; from
    `start_mapping_epoch` on it also trains/tests the global inference model with the FIXED mapping.
    `start_mapping_epoch` therefore means "first round the global model is trained".

Outputs in log_dir:
  rt_relation_table.csv   client-level table  (global_id, client_id, group_name, local_id, class_name)
  rt_mapping_acc.csv      mapping metrics, same column names as the repo's *_mapping_acc.csv
  rt_diagnostics.npz/json cos histograms, ties, per-class counts, CKKS-vs-plain bits, v-estimation error
  global_model_acc_<ds>.csv  Round, Epoch, Accuracy, Coverage  (unmapped samples counted WRONG)

exp_conf keys (all optional):
  rt_method              'filter'   'filter' | 'affscan' | 'attn' | 'attn_filter' | 'trivial'
                                    trivial: every client describes a class by the same agreed name
                                    ("3", "A", "car"); exact PSI on the names (label_mapping/psi_trivial_new.py)
  rt_psi                 'he'       'he' (CKKS, needs `pip install tenseal`) | 'plain' (debug)
                                    trivial: 'dh' (exact DH-PSI, default) | 'plain'
                                     | 'cpsi_helper' (BFV circuit-PSI + helper client; server sees final table only)
                                     | 'cpsi_2pc'    (VOLE circuit-PSI, per-pair 2PC, 2PC grouping; no helper)
                                     | 'cpsi_tag'    (VOLE circuit-PSI, equality tags; server groups in the clear)
                                     | 'psi_tag_hash' (per-check fuzzy-PSI tags, hash(text|image|verify); no circuit)
                                     cpsi_*: attn_filter/precision only
  rt_anchor_datasets     [FashionMNIST, USPS, CIFAR100]  public probes, TEST split; must not be used by any client
  rt_num_anchors         300
  rt_exemplars_per_class 64
  rt_min_samples         10         classes with fewer local samples skip pairing, still get own gid
  rt_coord               'alpha'    attn only: 'alpha' | 'coord'
  rt_ladder              null       list of thresholds; null -> 0.90..0.40 step 0.05
  rt_ladder_desc / rt_ladder_aff / rt_ladder_main   per-signal override (main = attn)
  rt_encoder_weights     'DEFAULT'  torchvision resnet18 weights; 'random' = explicit opt-in to random encoder
  rt_gt_aliases          {}         ground-truth only: {"car": "automobile", ...}
  rt_data_root           './data/raw'
  rt_max_classes         0          >0: each client randomly keeps <= N classes for RT (others: own gid)
  rt_only                false      true: stop after RT (diagnostics / mapping metrics only)
  rt_client_langs        [en, zh, es]   attn*: language each client writes its descriptions in (cycled by client id)
  rt_text_encoders       [paraphrase-multilingual-MiniLM-L12-v2]   attn*: each client's OWN text encoder
                                    (cycled by client id; may differ per client -- only the anchor LIST is shared)
  rt_descriptions        null       JSON {"<client_id>": {"<local_id>": ["kw1", "kw2", ...]}} overrides generated keywords
  rt_tau                 null       attn* temperature; null -> 0.05
  rt_group_by            'pacfl'    'pacfl' | 'dataset' (groups = one per dataset; for smoke tests / sanity)
"""
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from trainer.GeFL_gan_pacfl_iid.server import Server as OldServer
from label_mapping.label_mapping_utils import global_to_local_mapping
from label_mapping.rt_protocol import LADDER, to_group_map, pair_metrics, TEXT_ANCHORS
from label_mapping.circuit_psi_new import run_rt_protocol   # psi='circuit' -> server-free circuit-PSI
from label_mapping.rt_descriptions import keywords, LANGS
from label_mapping.psi_trivial_new import run_trivial      # rt_method: trivial (exact PSI on shared names)
from data.datasets import get_raw_dataset_transform

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _norm_name(s, aliases):
    s = str(s).strip()
    s = aliases.get(s, s)
    return s if len(s) == 1 else s.lower()      # single chars stay case-sensitive (EMNIST 'a' != 'A')


class Server(OldServer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # --label_mapping=rt   : RT protocol before training, fixed mapping, generator-only FL
        # any other method     : server.py behaviour (mapping at start_mapping_epoch) -- baselines,
        #                        but with the same test metric and the same new-client checkpoint
        self.rt_on = self.args.label_mapping == 'rt'
        self.rt_cfg = {k: v for k, v in self.exp_conf.items() if str(k).startswith('rt_')}
        self._cap_client_data(self.exp_conf.get('max_client_samples', 0))

    def _cap_client_data(self, n):
        """max_client_samples > 0: each client keeps a random subset of <= n training samples
        (same subset for every method, seeded). For small-scale runs (e.g. Mac); 0 = full data."""
        if not n:
            return
        from torch.utils.data import DataLoader, Subset
        for c in self.clients:
            ds = c.train_loader.dataset
            if len(ds) <= n:
                continue
            rng = np.random.default_rng((self.args.seed or 0) * 1000 + c.id)
            keep = sorted(rng.choice(len(ds), size=n, replace=False).tolist())
            if hasattr(ds, 'indices') and hasattr(ds, 'dataset'):          # keep Subset(dataset, idx) shape
                sub = Subset(ds.dataset, [ds.indices[k] for k in keep])
            else:
                sub = Subset(ds, keep)
            c.train_loader = DataLoader(sub, batch_size=c.train_loader.batch_size, shuffle=True, num_workers=0)
            c.num_samples = n
        self.logger.log(f"[Server] max_client_samples={n}: client training sets capped")

    def cfg(self, key, default):
        v = self.rt_cfg.get(key, default)
        return default if v is None else v

    def run(self):
        if self.cfg('rt_only', False):          # build + evaluate the relation table, skip FL training
            self.initialize_client_groups()
            self.logger.log("[RT] rt_only: true -> relation table done, FL training skipped")
            return
        super().run()
        self.save_newclient_checkpoint()

    def save_newclient_checkpoint(self):
        """server_checkpoints.pth + server_global_model.pth in the format trainer/NewClient/BaseNewClient
        expects: global_registry keyed by DATASET name (group map -> dataset map, majority per local id)."""
        votes = defaultdict(Counter)
        for c in self.clients:
            for a, g in self.local_id_to_global_id.get(c.group_name, {}).items():
                votes[(c.dataset_name, a)][g] += 1
        registry = defaultdict(dict)
        for (d, a), cnt in votes.items():
            registry[d][int(a)] = int(cnt.most_common(1)[0][0])
        n_global = len({g for mp in self.local_id_to_global_id.values() for g in mp.values()})
        ck = {
            'global_registry': dict(registry),
            'num_global_classes': n_global,
            'group_registry': self.local_id_to_global_id,
            'label_space_meta': {c.dataset_name: list(c.class_name_set) for c in self.clients},
            'global_feature_dim': self.global_feature_dim,
            'exp_conf': self.exp_conf,
            'args': {k: (str(v) if k == 'device' else v) for k, v in vars(self.args).items()},
        }
        torch.save(ck, os.path.join(self.logger.log_dir, 'server_checkpoints.pth'))
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(self.logger.log_dir, 'server_global_model.pth'))
        self.logger.log(f"[Server] new-client checkpoint saved: {self.logger.log_dir}/server_checkpoints.pth "
                        f"(+ server_global_model.pth), {n_global} global classes")

    def initialize_client_groups(self):
        if self.cfg('rt_group_by', 'pacfl') == 'dataset':
            self.client_groups = defaultdict(list)
            for c in self.clients:
                c.group_name = c.dataset_name
                self.client_groups[c.dataset_name].append(c)
            self.group_label_space_meta = {g: list(cs[0].class_name_set) for g, cs in self.client_groups.items()}
            self.logger.log(f"[RT] rt_group_by=dataset: groups = {sorted(self.client_groups)}")
        else:
            super().initialize_client_groups()   # PACFL clustering (needs group_name)
        # A group shares one generator (FedAvg) conditioned on LOCAL label id, so every client in it
        # must have the same label space. Mixed groups would also crash server.py at generator FedAvg
        # (embedding shapes differ, e.g. MNIST 10 vs EMNIST 62).
        for g, cs in self.client_groups.items():
            spaces = {tuple(c.class_name_set) for c in cs}
            if len(spaces) > 1:
                raise RuntimeError(
                    f"[RT] {g} mixes label spaces: clients {[c.id for c in cs]} "
                    f"datasets {sorted({c.dataset_name for c in cs})}. "
                    f"Lower --pacfl_cluster_alpha (more clusters) or set rt_group_by: dataset in the yaml.")
        if self.rt_on:
            self.rt_label_mapping()          # RT protocol once, before round 1

    # ------------------------------------------------------------------ public encoder
    def _rt_encoder(self):
        w = self.cfg('rt_encoder_weights', 'DEFAULT')
        if w == 'random':
            self.logger.log("[RT] WARNING: rt_encoder_weights=random (explicit). Mapping quality will be poor.")
            m = torchvision.models.resnet18(weights=None)
        else:
            try:
                m = torchvision.models.resnet18(weights=w)
            except Exception as e:           # E20: never fall back silently
                raise RuntimeError(
                    f"[RT] cannot load resnet18 weights '{w}': {e}\n"
                    f"Pre-download on this machine: python -c \"import torchvision; torchvision.models.resnet18(weights='{w}')\"\n"
                    f"or set rt_encoder_weights: random to opt in to a random encoder.") from e
        m.fc = nn.Identity()
        return m.to(self.device).eval()

    @torch.no_grad()
    def _embed(self, enc, x):
        x = x.to(self.device) * 0.5 + 0.5
        x = F.interpolate(x, size=64, mode='bilinear', align_corners=False)
        x = (x - _MEAN.to(self.device)) / _STD.to(self.device)
        return enc(x).cpu().numpy()

    def _anchors(self, enc):
        """B7: public probes from datasets NO client trains on (test split)."""
        names = list(self.cfg('rt_anchor_datasets', ['FashionMNIST', 'USPS', 'CIFAR100']))
        used = {c.dataset_name for c in self.clients}
        clash = used.intersection(names)
        if clash:
            raise ValueError(f"[RT] anchor datasets {sorted(clash)} are used by training clients; "
                             f"set rt_anchor_datasets to unused ones (clients use {sorted(used)})")
        n = self.cfg('rt_num_anchors', 300)
        root = self.cfg('rt_data_root', './data/raw')
        rng = np.random.default_rng(self.args.seed or 0)
        feats = []
        for k, name in enumerate(names):
            try:
                ds = get_raw_dataset_transform(name, root, train=False)
            except Exception as e:
                raise RuntimeError(f"[RT] anchor dataset {name} not found under {root} "
                                   f"(run data/prepare_dataset.py or change rt_anchor_datasets): {e}") from e
            per = n // len(names) + (k < n % len(names))
            idx = rng.choice(len(ds), size=per, replace=False)
            x = torch.stack([ds[int(t)][0] for t in idx])              # labels discarded
            feats.append(np.concatenate([self._embed(enc, x[s:s + 128]) for s in range(0, len(x), 128)]))
        self.logger.log(f"[RT] anchors: {n} from {names} (test split)")
        return np.concatenate(feats)

    def _rt_pick(self, client, count):
        """rt_max_classes: each client randomly keeps <= N classes (>= rt_min_samples) for RT;
        the rest skip pairing and get their own gid. Smoke-test / cost knob -- lowers recall."""
        k = self.cfg('rt_max_classes', 0)
        labels = sorted(count)
        if not k or len(labels) <= k:
            return labels
        ok = [l for l in labels if count[l] >= self.cfg('rt_min_samples', 10)]
        rng = np.random.default_rng((self.args.seed or 0) * 1000 + client.id)
        return sorted(int(v) for v in rng.choice(ok, size=min(k, len(ok)), replace=False))

    def _client_summary(self, enc, client, n_ex):
        """r_i(a) = mean encoder feature; n_ex=0 uses all samples in bounded batches.
        Fast path: read labels from Subset(dataset, indices).targets, load only the chosen exemplars.
        ponytail: computed server-side for simulation; in deployment this runs on the client."""
        if n_ex < 0:
            raise ValueError("rt_exemplars_per_class must be >= 0 (0 uses all samples)")
        ds = client.train_loader.dataset
        base = getattr(ds, 'dataset', None)
        tgt = next((getattr(base, a) for a in ('targets', 'labels') if hasattr(base, a)), None)
        if tgt is not None and hasattr(ds, 'indices'):
            y = np.asarray(tgt)[np.asarray(ds.indices)]
            count = Counter(y.tolist())
            summ = {}
            for l in self._rt_pick(client, count):
                pos = np.flatnonzero(y == l)[:n_ex or None]
                total = None
                for start in range(0, len(pos), 128):
                    x = torch.stack([ds[int(p)][0] for p in pos[start:start + 128]])
                    block_sum = self._embed(enc, x).sum(axis=0, dtype=np.float64)
                    total = block_sum if total is None else total + block_sum
                summ[int(l)] = total / len(pos)
            return summ, {int(k): v for k, v in count.items()}

        feats, got, count = {}, Counter(), Counter()      # fallback: streaming per-class sums
        for x, y in client.train_loader:
            y = y.numpy()
            count.update(y.tolist())
            keep = np.zeros(len(y), bool)
            for l in np.unique(y):
                idx = np.flatnonzero(y == l)[:max(n_ex - got[l], 0) if n_ex else None]
                keep[idx] = True
                got[l] += len(idx)
            if keep.any():
                f = self._embed(enc, x[torch.from_numpy(keep)])
                for l, v in zip(y[keep], f):
                    if l not in feats:
                        feats[l] = np.zeros_like(v, dtype=np.float64)
                    feats[l] += v
        return {int(l): v / got[l] for l, v in feats.items()}, {int(k): v for k, v in count.items()}

    def _client_counts(self, client):
        """Local class counts only (rt_method: trivial needs no encoder)."""
        ds = client.train_loader.dataset
        base = getattr(ds, 'dataset', None)
        tgt = next((getattr(base, a) for a in ('targets', 'labels') if hasattr(base, a)), None)
        if tgt is not None and hasattr(ds, 'indices'):
            count = Counter(np.asarray(tgt)[np.asarray(ds.indices)].tolist())
        else:
            count = Counter()
            for _, y in client.train_loader:
                count.update(y.numpy().tolist())
        return {int(k): v for k, v in count.items()}

    # ------------------------------------------------------------------ attn*: client-written descriptions
    def _rt_text_side(self, clients, rows):
        """Each client writes KEYWORDS for each own class in its own language (e.g. ["數字", "三"]) and
        encodes every keyword SEPARATELY, plus the public anchor LIST, with ITS OWN text encoder.
        keyword_attn in rt_protocol then attends each keyword over the anchors and weights them by
        local IDF (a keyword shared by all own classes, like "digit" on an MNIST client, gets weight 0).
        Nothing but TEXT_ANCHORS is agreed.
        ponytail: simulated server-side; in deployment each client does this locally."""
        from sentence_transformers import SentenceTransformer
        langs = list(self.cfg('rt_client_langs', ['en', 'zh', 'es']))
        encs = list(self.cfg('rt_text_encoders', ['paraphrase-multilingual-MiniLM-L12-v2']))
        for l in langs:
            if l not in LANGS:
                raise ValueError(f"rt_client_langs: {l} not in {LANGS}")
        override = {}
        if self.cfg('rt_descriptions', None):
            with open(self.cfg('rt_descriptions', None), encoding='utf-8') as f:
                override = {int(k): {int(a): ([t] if isinstance(t, str) else list(t)) for a, t in v.items()}
                            for k, v in json.load(f).items()}
        models = {}
        for c in self.clients:
            if c.id not in clients:
                continue
            lang, enc_name = langs[c.id % len(langs)], encs[c.id % len(encs)]
            if enc_name not in models:
                self.logger.log(f"[RT] loading text encoder {enc_name}")
                models[enc_name] = SentenceTransformer(enc_name, device='cpu')
            m = models[enc_name]
            labels = sorted(clients[c.id]["summ"])
            kws = [override.get(c.id, {}).get(a) or keywords(c.dataset_name, c.class_name_set[a], lang) for a in labels]
            vocab = list(dict.fromkeys(k for kw in kws for k in kw))
            vec = dict(zip(vocab, m.encode(vocab, normalize_embeddings=True)))
            clients[c.id]["keywords"] = dict(zip(labels, kws))
            clients[c.id]["keyword_vecs"] = {a: np.stack([vec[k] for k in kw]) for a, kw in zip(labels, kws)}
            clients[c.id]["anchor_vecs"] = m.encode(TEXT_ANCHORS, normalize_embeddings=True)
            rows += [(c.id, a, lang, enc_name, " | ".join(kw)) for a, kw in zip(labels, kws)]
        self.logger.log(f"[RT] descriptions: langs={langs} encoders={encs} anchors={len(TEXT_ANCHORS)} words")

    # ------------------------------------------------------------------ RT protocol
    def rt_label_mapping(self):
        method = self.cfg('rt_method', 'filter')
        psi = self.cfg('rt_psi', 'dh' if method == 'trivial' else 'he')
        self.logger.log(f"[RT] Building relation table before training: method={method}, psi={psi}")
        if method == 'trivial':
            return self._rt_trivial(psi)
        enc = self._rt_encoder()
        P = self._anchors(enc)
        mu = P.mean(axis=0)                  # public head: subtract public anchor mean
        P = P - mu
        match = self.cfg('rt_attn_match', 'precision')
        n_ex = self.cfg('rt_exemplars_per_class', 0 if method == 'attn_filter' and match == 'precision' else 64)

        clients = {}
        for c in tqdm(self.clients, desc="[RT] Phase 1 local tables"):
            summ, count = self._client_summary(enc, c, n_ex)
            clients[c.id] = {"summ": {a: r - mu for a, r in summ.items()},
                             "names": {a: c.class_name_set[a] for a in summ},
                             "count": count}
        del enc
        torch.cuda.empty_cache()

        desc_rows = []
        if method in ('attn', 'attn_filter'):
            self._rt_text_side(clients, desc_rows)

        aliases = self.cfg('rt_gt_aliases', {})
        by_id = {c.id: c for c in self.clients}
        name = lambda i, a: _norm_name(by_id[i].class_name_set[a], aliases)
        gt = lambda i, a, j, b: name(i, a) == name(j, b)                 # A4: explicit relation fn

        masked = {}
        base = self.cfg('rt_ladder', LADDER.tolist())
        ladder = {k: [float(v) for v in self.cfg(f'rt_ladder_{k}', base)] for k in ('desc', 'aff', 'main')}
        table, edges, diag = run_rt_protocol(
            clients, P, method=method, psi=psi, ladder=ladder,
            tau=self.cfg('rt_tau', None), coord=self.cfg('rt_coord', 'log_whiten'),
            attn_match=match, verify_floor=self.cfg('rt_verify_floor', .90),
            verify_margin=self.cfg('rt_verify_margin', .01),
            min_samples=self.cfg('rt_min_samples', 10), seed=self.args.seed or 0,
            gt=gt, log=self.logger.log, masked_out=masked)

        self._rt_finish(method, psi, clients, table, edges, diag, masked, desc_rows, ladder, len(P), gt, aliases)

    def _rt_trivial(self, psi):
        """rt_method: trivial -- each client's description of class a is the agreed common name
        (shared_name: '3', 'A', 'car'); exact PSI on those names gives the relation table."""
        aliases = self.cfg('rt_gt_aliases', {})
        clients = {}
        for c in self.clients:
            count = self._client_counts(c)
            clients[c.id] = {"names": {a: c.class_name_set[a] for a in self._rt_pick(c, count)}, "count": count}
        by_id = {c.id: c for c in self.clients}
        name = lambda i, a: _norm_name(by_id[i].class_name_set[a], aliases)
        gt = lambda i, a, j, b: name(i, a) == name(j, b)
        masked = {}
        # names need no image summary -> every held class joins the PSI (rt_min_samples is for fuzzy methods)
        table, edges, diag = run_trivial(clients, psi=psi, min_samples=self.cfg('rt_trivial_min_samples', 1),
                                         seed=self.args.seed or 0, aliases=aliases, log=self.logger.log,
                                         masked_out=masked)
        desc_rows = [(i, a, "", "exact", name(i, a)) for i, c in clients.items() for a in c["names"]]
        self._rt_finish('trivial', psi, clients, table, edges, diag, masked, desc_rows, None, 0, gt, aliases)

    def _rt_finish(self, method, psi, clients, table, edges, diag, masked, desc_rows, ladder, n_anchors, gt, aliases):
        by_id = {c.id: c for c in self.clients}
        group_of = {c.id: c.group_name for c in self.clients}
        mapping = to_group_map(table, group_of)
        self.local_id_to_global_id = mapping
        global_to_local_mapping(mapping, logger=self.logger, label_space_meta=self.group_label_space_meta)

        # ---- metrics (client level = what the protocol produced; group level = what training uses)
        m_client = pair_metrics(table, lambda k1, k2: gt(k1[0], k1[1], k2[0], k2[1]))
        gnames = self.group_label_space_meta
        g_assign = {(g, a): v for g, mp in mapping.items() for a, v in mp.items()}
        m_group = pair_metrics(g_assign, lambda k1, k2: _norm_name(gnames[k1[0]][k1[1]], aliases)
                               == _norm_name(gnames[k2[0]][k2[1]], aliases))
        for lvl, m in (("client", m_client), ("group", m_group)):
            self.logger.log(f"[RT] {lvl:6s} F1={m['f1_score']:.4f} MCC={m['mcc']:.4f} "
                            f"P={m['precision']:.4f} R={m['recall']:.4f} (TP={m['TP']} FP={m['FP']} FN={m['FN']})")

        d = self.logger.log_dir
        with open(os.path.join(d, "rt_mapping_acc.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            cols = ["recall", "specificity", "precision", "average_accuracy", "f1_score", "mcc", "TP", "FP", "TN", "FN"]
            w.writerow(["level", "method", "psi", "n_edges", "n_global_ids"] + cols)
            for lvl, m in (("client", m_client), ("group", m_group)):
                w.writerow([lvl, method, psi, len(edges), len(set(table.values()))] + [m[c] for c in cols])

        with open(os.path.join(d, "rt_local_tables.csv"), 'w', newline='') as f:     # tex: |id|masked_id|description|
            w = csv.writer(f)
            w.writerow(["client_id", "group_name", "local_id", "masked_id", "lang", "text_encoder", "keywords"])
            dmap = {(r[0], r[1]): r for r in desc_rows}
            for (cid, a), mid in sorted(masked.items()):
                r = dmap.get((cid, a), (cid, a, "", "", by_id[cid].class_name_set[a]))
                w.writerow([cid, group_of[cid], a, mid, r[2], r[3], r[4]])

        with open(os.path.join(d, "rt_relation_table.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["global_id", "client_id", "group_name", "local_id", "class_name"])
            for (cid, a), g in sorted(table.items(), key=lambda t: (t[1], t[0])):
                w.writerow([g, cid, group_of[cid], a, by_id[cid].class_name_set[a]])

        counts = np.array([n for c in clients.values() for n in c["count"].values()])
        min_s = self.cfg('rt_trivial_min_samples', 1) if method == 'trivial' else self.cfg('rt_min_samples', 10)
        np.savez_compressed(os.path.join(d, "rt_diagnostics.npz"), class_counts=counts,
                            **{k: v for k, v in diag.items() if np.ndim(v) > 0})
        summary = {k: int(v) for k, v in diag.items() if np.ndim(v) == 0}
        summary.update(method=method, psi=psi, ladder=ladder, n_anchors=int(n_anchors),
                       classes_below_min=int((counts < min_s).sum()),
                       metrics_client=m_client, metrics_group=m_group)
        with open(os.path.join(d, "rt_diagnostics.json"), 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        self.logger.log(f"[RT] saved rt_mapping_acc.csv, rt_relation_table.csv, rt_diagnostics.* to {d}")

    # ------------------------------------------------------------------ training: generator FL only
    def aggregate(self):
        if not self.rt_on:                   # baseline: original mapping at start_mapping_epoch
            return OldServer.aggregate(self)
        groups = defaultdict(list)
        for client in self.selected_clients:
            groups[client.group_name].append(client)
            self.label_space_meta.setdefault(client.group_name, client.class_name_set)
        for g, cs in groups.items():
            self.global_gen_states[g] = self.aggregate_weights(
                [(c.num_samples, c.generator.state_dict()) for c in cs])
            self.global_dis_states[g] = self.aggregate_weights(
                [(c.num_samples, c.discriminator.state_dict()) for c in cs])
        if (self.glob_iter + 1) >= self.start_mapping_epoch:
            self.train_global_inference_model()
            self.test_global_inference_model()

    # ------------------------------------------------------------------ C11b: unmapped = wrong
    @torch.no_grad()
    def test_global_inference_model(self, epoch=None):
        if self.model is None:
            return
        self.model.eval().to(self.device)
        correct, total, mapped = Counter(), Counter(), Counter()
        for client in self.clients:
            d_name = client.dataset_name
            mp = self.local_id_to_global_id.get(client.group_name, {})
            for x, y in client.test_loader:
                _, logits = self.model(x.to(self.device))
                pred = logits.argmax(1).cpu().numpy()
                for p, t in zip(pred, y.numpy()):
                    total[d_name] += 1
                    if int(t) in mp:
                        mapped[d_name] += 1
                        correct[d_name] += int(p == mp[int(t)])
        rows = {}
        for d_name in sorted(total):
            acc = 100.0 * correct[d_name] / total[d_name]
            cov = 100.0 * mapped[d_name] / total[d_name]
            rows[d_name] = (acc, cov)
            self.logger.log(f"[Server] Global Acc ({d_name}) R{self.glob_iter + 1}: {acc:.2f}% | coverage {cov:.2f}%")
        T = sum(total.values())
        rows["mix"] = (100.0 * sum(correct.values()) / T, 100.0 * sum(mapped.values()) / T)
        for d_name, (acc, cov) in rows.items():
            path = os.path.join(self.logger.log_dir, f"global_model_acc_{d_name}.csv")
            new = not os.path.isfile(path)
            with open(path, 'a', newline='') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["Round", "Epoch", "Accuracy", "Coverage"])
                w.writerow([self.glob_iter + 1, epoch, round(acc, 2), round(cov, 2)])
