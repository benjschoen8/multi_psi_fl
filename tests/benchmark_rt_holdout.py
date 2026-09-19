"""Mapping-only holdout checks using real MNIST prototypes, no classifier training.

Run from the repo root:
HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tests/benchmark_rt_holdout.py

Seed 17 images and random split seed 2026 are separate from development seeds 0-2.
All variants receive the same samples, encoders and anchor banks. This checks
new MNIST subsets, not generalization to a different dataset or language family.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import test_mnist_split_new as harness
from label_mapping.rt_protocol import LADDER, run_rt_protocol, to_group_map, pair_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=100)
    parser.add_argument('--image-seed', type=int, default=31)
    parser.add_argument('--split-seed', type=int, default=731)
    benchmark_args = parser.parse_args()
    per_class = benchmark_args.per_class
    sys.argv = [sys.argv[0], '--name_style', 'multilang', '--descriptions', 'keywords',
                '--desc_encoder', 'st:paraphrase-multilingual-MiniLM-L12-v2']
    args = harness.parse()
    ds = harness.get_raw_dataset_transform('MNIST', args.data_root, train=True)
    rng = np.random.default_rng(benchmark_args.image_seed)
    pool = {d: list(rng.permutation(np.flatnonzero(np.asarray(ds.targets) == d))) for d in range(10)}
    parties, cid = [], 0
    for k in range(3):
        digits = list(range(10))
        p = {'name': f'P{k}', 'digits': digits,
             'names': harness.party_names(k, digits, 'multilang'), 'clients': []}
        for _ in range(2):
            idx = [pool[d].pop() for d in digits for _ in range(per_class)]
            data = harness.Relabel(ds, idx, dict(zip(digits, digits)))
            p['clients'].append({'cid': cid, 'data': [data[n] for n in range(len(data))]})
            cid += 1
        parties.append(p)
    clients, owner, P, K = harness.psi_inputs(args, parties, 'cpu', benchmark_args.image_seed, True)
    print(f'Prepared seed-{benchmark_args.image_seed} image and text embeddings.', flush=True)
    splits = [('all_digits', [list(range(10))] * 3),
              ('disjoint', [[0, 1, 2], [3, 4, 5], [6, 7, 8]])]
    rng = np.random.default_rng(benchmark_args.split_seed)
    for n in range(20):
        splits.append((f'random_{n}', [sorted(rng.choice(10, size=int(rng.integers(3, 8)), replace=False).tolist())
                                       for _ in range(3)]))
    variants = {
        'old_attn': dict(method='attn', coord='alpha', attn_confidence='positive'),
        'new_attn': dict(method='attn'),
        'previous_fusion': dict(method='attn_filter', attn_match='fusion'),
        'previous_fusion_all': dict(method='attn_filter', attn_match='fusion'),
        'precision': dict(method='attn_filter', attn_match='precision'),
        'precision_64': dict(method='attn_filter', attn_match='precision'),
        'psi_filter': dict(method='filter'),
        'psi_affscan': dict(method='affscan'),
        'image_only_confident': dict(method='affscan'),
    }
    ladder = {'main': LADDER, 'desc': np.round(np.arange(.9, .449, -.05), 2),
              'aff': [.99, .985, .98, .975, .97, .965, .96, .955, .95]}
    rows = []
    for name, subsets in splits:
        selected = {}
        for i, c in clients.items():
            labels = subsets[int(owner[i][1:])]
            selected[i] = {k: ({a: v[a] for a in labels} if isinstance(v, dict) else v)
                           for k, v in c.items()}
        for variant, options in variants.items():
            run_clients = selected
            if variant == 'previous_fusion_all':
                run_clients = {i: {**c, 'summ': c['precision_summ']} for i, c in selected.items()}
            elif variant == 'precision_64':
                run_clients = {i: {**c, 'precision_summ': c['summ']} for i, c in selected.items()}
            run_ladder = {**ladder, 'aff': [.99]} if variant == 'image_only_confident' else ladder
            table, _, _ = run_rt_protocol(run_clients, P, K_text=K, psi='plain', ladder=run_ladder,
                                           log=lambda *_: None, min_samples=0, **options)
            mapping = to_group_map(table, owner)
            assign = {(i, a): gid for i, mp in mapping.items() for a, gid in mp.items()}
            # Here local ids equal digits, only for evaluation after protocol completes.
            metrics = pair_metrics(assign, lambda a, b: a[1] == b[1])
            rows.append({'split': name, 'subsets': subsets, 'per_class': per_class,
                         'image_seed': benchmark_args.image_seed, 'split_seed': benchmark_args.split_seed,
                         'variant': variant, **metrics})
        print(name, {v: round(rows[-len(variants) + n]['f1_score'], 3)
                     for n, v in enumerate(variants)}, flush=True)
    output = Path(f'logs/rt_precision_holdout_{per_class}_{benchmark_args.image_seed}_{benchmark_args.split_seed}.json')
    output.write_text(json.dumps(rows, indent=2))
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
