# RT attention revision: implementation and validation

The revised protocol improves the tested multilingual MNIST alignment; it is not a guarantee of superiority across datasets. Development inspected the original split and seeds 0–2. Final training covers seeds 0–4. Additional subset tests use image seed 17 and split seed 2026, held fixed after selecting the rule.

## What changed

1. Keep individually encoded keywords and local IDF aggregation. No class translations, number anchors, learned projection layers, or ground-truth weights were added.
2. Replace centered softmax probabilities with centered log-attention corrected for the public anchor Gram matrix. If `Kc = U diag(s) V.T`, use `W = U diag(1/s) U.T` with singular values below `1e-6 * max(s)` removed. The PSI signal is `Unit(center(log(alpha)) W)`. This retains anchor coordinates while reducing redundancy and peaked-softmax distortion. With one active keyword this is an anchor-span projection, not learned keyword selection. Arbitrarily different encoders are not guaranteed to align; tests establish invariance to local orthogonal rotations and zero-padded dimensions.
3. Require strong evidence for open-set matching. Text-only matches must exceed the highest configured text ladder threshold (0.90 in this benchmark). Mixed matches require positive text and image ranks, and at least one modality must exceed its highest threshold (text 0.90 or image 0.99). These are existing ladder endpoints, not oracle-dependent thresholds.
4. For `attn_filter`, combine normalized PSI ranks before mutual matching: the integer score is `text_rank * image_rungs + image_rank * text_rungs`. This lets images recover text-ambiguous candidates, while two weak signals cannot force an unmatched class to merge. Exact rank ties are still rejected; no plaintext similarity tie-break was added.

The first unguarded fusion trial increased recall but created false merges and reduced strict accuracy. Its results are retained in `logs/rt_attention_fix/2026-09-19_23-38-01/`; they were rejected rather than presented as the final outcome.

## Five-seed end-to-end benchmark

Original default split, two clients per party, 1,000 samples/client, 3 local epochs, 5 global classifier epochs, the original multilingual MiniLM encoder, original public anchors and ladders. GPU MPS, PSI plaintext reference for this full run. Baseline implementations were unchanged during this revision. Standard deviations below are population standard deviations, not confidence intervals.

| Method | Mapping F1 (mean ± std) | Strict accuracy % (mean ± std) | Digit accuracy % (mean ± std) |
|---|---:|---:|---:|
| psi_attn | 0.857 ± 0.000 | 78.32 ± 7.87 | 84.77 ± 8.87 |
| psi_attn_filter | 1.000 ± 0.000 | 76.96 ± 8.52 | 76.96 ± 8.52 |
| psi_filter | 0.771 ± 0.043 | 65.10 ± 6.40 | 76.12 ± 9.82 |
| image-bi | 0.686 ± 0.093 | 74.32 ± 9.22 | 86.23 ± 6.37 |
| oracle | 1.000 ± 0.000 | 76.96 ± 8.52 | 76.96 ± 8.52 |

The mixed method's group assignments equal oracle on all five seeds; its global classifier accuracy therefore also equals oracle. Pure attention still misses one of the four true cross-party matches. Strict accuracy can exceed oracle because different mappings train different output spaces; oracle is an alignment reference, not a finite-training accuracy upper bound. These five seeds do not establish statistical significance.

Full artifacts: `logs/rt_attention_fix_final/2026-09-19_23-42-03/` (configuration, per-seed metrics, summaries, mappings and PSI diagnostics).

## Additional mapping-only subset checks

One fixed seed-17 image bank, English/Chinese/Spanish metadata, two clients per party. Twenty random partially overlapping subsets, generated with seed 2026, plus all-digit and disjoint controls. The subsets share data and are not twenty independent training seeds. The 100-sample setting still uses the harness's existing maximum of 64 images per prototype. No classifier training is included in these checks.

Mean mapping F1 across the 20 random subsets:

| Method | 5 images/class/client | 100 images/class/client |
|---|---:|---:|
| old_attn | 0.129 | 0.129 |
| new_attn | 0.761 | 0.761 |
| new_attn_filter | 0.879 | 0.996 |
| psi_filter | 0.628 | 0.724 |
| psi_affscan | 0.620 | 0.864 |
| image_only_confident | 0.413 | 0.987 |

`image_only_confident` is an ablation that applies the same 0.99 strong-image threshold without text. It is competitive with plentiful images; this prevents attributing all gains to attention. The clearest complementary benefit appears with only five images per class. The mixed method is not best on every subset (e.g. five-image `random_17`). Disjoint controls have no true matches: interpret false-positive counts, not F1=0. Both new methods produce zero false positives there. These checks remain within MNIST and these three languages.

## Verification and cost

- Ten regression tests cover independent keyword encoding, local IDF, duplicate keywords, single-class fallback, rotation/dimension invariance, projection geometry, confidence rejection, image rescue, ground-truth independence, and plaintext/CKKS agreement for both new methods.
- An additional real seed-0 check with one client per language gave identical plaintext/CKKS tables and edges and zero rank mismatches for `attn` and `attn_filter`; see `logs/rt_attention_he_validation.json`.
- The original protocol self-check also passes for filter, affscan, attn and attn_filter in both plaintext and HE modes.
- The full five-seed accuracy benchmark used plaintext PSI; it is not a HE runtime benchmark.
- Fusion requests image ranks for every positive text pair, rather than only classes referenced by mutually selected text candidates. This can increase encrypted work and reveals more image ranks. It is a protocol change from the original Alg.5 cascade and must be reflected in any privacy/leakage discussion.
- IDF is a fixed metadata rule. In this MNIST experiment the shared domain keyword has zero weight, so the performance gain is from the coordinate correction, confidence rejection, and cross-modal evidence—not learned keyword selection.

## Reproduce

Run from the repository root:

```sh
HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -u test_mnist_split_new.py --device mps --seeds 5 --name_style multilang --descriptions keywords --desc_encoder st:paraphrase-multilingual-MiniLM-L12-v2 --methods psi_filter,psi_attn,psi_attn_filter,image-bi,oracle --psi plain --attn_coord log_whiten --attn_match fusion --attn_confidence top --out logs/rt_attention_fix_final
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tests -p 'test_rt_keyword_attention.py'
HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tests/benchmark_rt_holdout.py --per-class 5
HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tests/benchmark_rt_holdout.py --per-class 100
HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 python3 tests/benchmark_keyword_attention.py
```

Cached models and datasets are required for offline runs. `--psi he` selects encryption. The defaults are now `--attn_coord log_whiten --attn_heads 1 --attn_match fusion --attn_confidence top`. To reproduce the prior keyword protocol, explicitly use `--attn_coord alpha --attn_match intersection --attn_confidence positive`. Existing callers that explicitly supply `coord="alpha"` keep that coordinate setting; unrelated training configurations were not rewritten. The legacy `intersection` path always uses its original positive-rank cascade.
