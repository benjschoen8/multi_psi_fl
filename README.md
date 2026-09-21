# Multi-FL with a PSI relation table

Model-heterogeneous federated learning over several datasets (MNIST, EMNIST, CIFAR-10) whose clients use
**different, private label ids**. Before training, clients run a **relation-table (RT) protocol** based on
fuzzy private set intersection (PSI) to decide which local classes are the same class. The server only
ever sees masked ids. Training then does generator-based FL (GeFL + PACFL clustering) with that fixed mapping.

Baseline mappings from the original code (image-bi, missing link, feature-bi, cosine similarity) run in the
same pipeline, so the resulting plots compare like with like.

## Contents

- [Quick start](#quick-start)
- [Experiments](#experiments)
- [RT methods](#rt-methods)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Repository layout](#repository-layout)
- [Notes and limitations](#notes-and-limitations)

## Quick start

```bash
# 1. dependencies (Python 3.9+)
python3 -m pip install torch torchvision numpy pandas matplotlib omegaconf tqdm tenseal sentence-transformers

# 2. datasets -> data/raw/  (MNIST, EMNIST, CIFAR10 for training; FashionMNIST, USPS, CIFAR100 as public anchors)
python3 data/prepare_dataset.py

# 3. pre-download the two pretrained models (the run fails loudly instead of silently using random weights)
python3 -c "import torchvision; torchvision.models.resnet18(weights='DEFAULT')"
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# 4. run the main comparison (Apple-silicon GPU, small scale)
bash run_mac_global_acc_new.sh                       # original settings: 45 rounds, all global models from round 25, 10 clients/dataset, all data
bash run_mac_global_acc_new.sh 25 15 3 mps "" 2000   # Mac-sized: 25 rounds, all from round 15, 3 clients, 2000 samples/client
```

The plots land in `plot/global_accuracy_plots/mac_r10_s5_c3_psi/`.

## Experiments

| Goal | Command | Hardware |
|---|---|---|
| Global-model accuracy: 7 PSI versions vs 4 baselines | `bash run_mac_global_acc_new.sh [ROUNDS=45] [START=25] [CLIENTS_PER_DATASET=10] [DEVICE=auto] [WARMUP=START-1] [CAP=0]` | any (cuda / mps / cpu) |
| Global accuracy + mapping metrics, full scale | `bash run_rt_new.sh [START=25] [DEVICE=cuda:0]` | NVIDIA / ROCm GPU |
| New-client fine-tuning comparison | `bash run_newclient_psi_new.sh [START=25] [DEVICE=cuda:0] [EPOCHS=30]` | NVIDIA / ROCm GPU |
| Mapping-only test on MNIST digit subsets (no FL, minutes) | `python3 test_mnist_split_new.py --device mps --seeds 5` | any |
| Unit tests for the attention RT | `python3 -m unittest discover -s tests -p 'test_rt_keyword_attention.py'` | CPU |

- `START` is the round the baselines build their mapping and start training the global model; they need
  trained generators first. PSI builds its mapping before round 1 and trains the global model from round 1.
- The scripts skip runs whose logs are already complete, so an interrupted script can simply be rerun.

### Single run

```bash
python3 main_new.py --seed=15698 --algorithm=GeFL_gan_pacfl_iid \
  --num_train_mnist=3 --num_train_emnist=3 --num_train_cifar10=3 \
  --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --num_new_clients=1 \
  --device=mps --pacfl_cluster_alpha=20 --pacfl_basis_budget=20 --start_mapping_epoch=1 \
  --label_mapping=rt --exp_conf=./configs/het-iid-exp_rt_attn_filter_mps_new.yaml --exp_timestamp=my_run
```

`--label_mapping=rt` selects the PSI protocol, with the method set by `rt_method` in the yaml. Any other value
(`image-bi`, `missing_link`, `feature-bi`, `image-cs`, ...) runs the original baseline through the same server.

### MNIST split test

Parties get overlapping digit subsets, e.g. `0,1,2,3,4 ; 5,6,7,8 ; 0,1,5,6,9`, each with its own local ids.
The test reports mapping F1 and global-model accuracy.

```bash
python3 test_mnist_split_new.py --device mps --seeds 5                        # names are digits
python3 test_mnist_split_new.py --device mps --seeds 5 --per_class 20         # data-scarce
python3 test_mnist_split_new.py --device mps --seeds 5 --name_style multilang \
  --desc_encoder st:paraphrase-multilingual-MiniLM-L12-v2                     # en / zh / es descriptions
```

Useful flags: `--splits`, `--methods psi_attn,psi_attn_filter,image-bi,oracle`, `--psi plain` (no encryption).

## RT methods

`rt_method` in the yaml, or `psi_<method>` in the MNIST test.

| Method | Description signal | Image signal | tex algorithm |
|---|---|---|---|
| `filter` | class-name / description vectors → fuzzy PSI → candidates | image-affinity PSI filters the candidates | Alg. 2 + Alg. 5 |
| `affscan` | — | affinity profile over public image probes | Alg. 3 |
| `attn` | client-written description → attention over a public anchor word list → log-whitened coordinates | — | Alg. 4 |
| `attn_filter` | same as `attn` | image ranks and verification (`rt_attn_match`: `precision` / `fusion` / `intersection`) | Alg. 4 + Alg. 5 |

In the `attn*` methods, each client writes descriptions of its own classes, in its own language
(`rt_client_langs`), with its own text encoder (`rt_text_encoders`). It embeds the public anchor word list
(`TEXT_ANCHORS` in `label_mapping/rt_protocol.py`) with that same encoder. The ordered anchor list is the only
thing agreed in advance.

All methods then build global ids with union–find over masked ids: a group never holds two classes of the
same client.

**PSI.** `rt_psi: he` runs CKKS through TenSEAL. The receiver keeps the secret key and the sender only sees the
public context; only the ladder ranks are revealed. `rt_psi: plain` returns the same bits without
encryption, for debugging.

## Configuration

Base config: `configs/het-iid-exp.yaml`. The `*_new.yaml` files add these RT keys:

| Key | Default | Meaning |
|---|---|---|
| `rt_method` | `filter` | `filter` / `affscan` / `attn` / `attn_filter` |
| `rt_psi` | `he` | `he` (CKKS) or `plain` |
| `rt_anchor_datasets` | `[FashionMNIST, USPS, CIFAR100]` | public image probes (test split, used by no client) |
| `rt_num_anchors` | `300` | number of image probes |
| `rt_exemplars_per_class` | `64` (`0` = all) | images per class for the class prototype |
| `rt_min_samples` | `10` | classes with fewer samples skip matching and get their own id |
| `rt_ladder`, `rt_ladder_desc`, `rt_ladder_aff`, `rt_ladder_main` | 0.90 → 0.40 | PSI threshold ladders, global or per signal |
| `rt_client_langs` | `[en, zh, es]` | `attn*`: description language per client (cycled) |
| `rt_text_encoders` | `[paraphrase-multilingual-MiniLM-L12-v2]` | `attn*`: text encoder per client (cycled) |
| `rt_descriptions` | none | JSON `{"<client_id>": {"<local_id>": "text"}}` overriding the generated descriptions |
| `rt_tau`, `rt_coord`, `rt_attn_match` | `0.05`, `log_whiten`, `precision` | attention temperature, coordinate type, text/image combination |
| `rt_group_by` | `pacfl` | `dataset` = one group per dataset (use for small client counts) |
| `rt_only` | `false` | stop after building the relation table |
| `rt_max_classes` | `0` | smoke-test knob: each client matches at most N random classes |
| `max_client_samples` | `0` | cap each client's training set (small-scale runs) |

## Outputs

Each run writes to `logs/<exp_timestamp>/GeFL_gan_pacfl_iid/`:

| File | Content |
|---|---|
| `global_model_acc_<dataset>.csv`, `global_model_acc_mix.csv` | per-round global accuracy and coverage (unmapped test samples count as wrong) |
| `rt_relation_table.csv` | global id ↔ (client, local id, class name) |
| `rt_local_tables.csv` | each client's local table: local id, masked id, language, encoder, description |
| `rt_mapping_acc.csv` | mapping precision / recall / F1 / MCC at client and group level |
| `rt_diagnostics.npz/json` | cosine histograms, ties, PSI leakage and HE-vs-plaintext checks |
| `server_checkpoints.pth`, `server_global_model.pth` | checkpoint for the new-client experiments |

Plots:

```bash
python3 plot/plot_global_model_acc_new.py --psi "PSI=logs/<run>/GeFL_gan_pacfl_iid" \
  --run "Ours (image-bi)=logs/<run2>/GeFL_gan_pacfl_iid" --datasets MNIST EMNIST CIFAR10 --out plot/my_plot
python3 plot/plot_rt_diagnostics_new.py logs/<run>/GeFL_gan_pacfl_iid
```

## Repository layout

```
main.py / main_new.py            entry point (main_new adds --label_mapping=rt, loads server_new.py)
trainer/GeFL_gan_pacfl_iid/      server.py (original) + server_new.py (RT before training, generator-only FL)
trainer/BaseFL, trainer/NewClient
label_mapping/rt_protocol.py     RT protocol: fuzzy PSI (CKKS), filter / affscan / attn / attn_filter, union-find
label_mapping/rt_descriptions.py simulated client-written descriptions in 6 languages
label_mapping/label_mapping_utils.py  original baseline mappings
test_mnist_split_new.py          mapping-only MNIST subset test
tests/                           unit tests and benchmarks for the attention RT
plot/                            plotting scripts (*_new.py)
utils/plot_comparison_new.py     new-client comparison plot
configs/                         base config + RT configs (*_new.yaml)
data/                            dataset loading and partitioning (data/raw is git-ignored)
docs/rt_attention_revision.md    notes on the attention revision
```

## PSI method names

All PSI variants build the relation table once, before training. `rt_method` picks the signals,
`rt_psi` (or `--psi` in the split test) picks the protocol.

| name | `rt_method` / `rt_psi` | signals | who learns matches |
|---|---|---|---|
| FPSI-DescFilter [CKKS] | `filter` / `he` | description, then image filter | receiver of each pair |
| FPSI-AttnImage [CKKS] | `attn_filter` / `he` | keyword attention + image (precision rule) | receiver of each pair |
| FPSI-AttnText [CKKS] | `attn` / `he` | keyword attention only | receiver of each pair |
| CPSI-Helper [BFV] | `attn_filter` / `cpsi_helper` | as AttnImage, one PSI message | nobody; server gets final table (helper client deals triples) |
| CPSI-2PC [VOLE] | `attn_filter` / `cpsi_2pc` | as AttnImage | nobody; server gets final table (no helper) |
| CPSI-Tag [VOLE] | `attn_filter` / `cpsi_tag` | as AttnImage | nobody; server gets equality tags = pairwise matches, groups itself |
| PSI-TagHash [OPPRF] | `attn_filter` / `psi_tag_hash` | text, image, verify thresholds (no rival margin) | nobody; server gets hash(tag_text\|tag_image\|tag_verify) = AND bits per branch |

`plain` = same as `he` without crypto (debugging). `circuit`, `vole`, `vole_tag` are accepted as old aliases.

```bash
python3 tests/test_circuit_psi_new.py
python3 test_mnist_split_new.py --device mps --seeds 5 --methods psi_attn_filter --psi cpsi_tag
```

FL configs: `configs/het-iid-exp_rt_cpsi_{helper,2pc,tag}_mps_new.yaml`, `configs/het-iid-exp_rt_psi_tag_hash_mps_new.yaml`.

## Notes and limitations

- **Simulation.** All parties run in one process; the server computes client summaries for convenience. The
  cryptography is real CKKS, but there is no network.
- **Security model.** Semi-honest. With `rt_psi: he` the receiver sees ladder ranks and multiplicatively masked
  magnitudes. `rt_attn_match: precision` adds a fine verification ladder over raw image features, which
  reveals more. Both are quantified in `rt_diagnostics` (`v_est_err`).
- **Memory.** The first run downloads about 470 MB for the multilingual text model.
- **Mac.** Use `--device=mps` with `PYTORCH_ENABLE_MPS_FALLBACK=1`. LeNet and AlexNet (client model ids 6 and 7)
  do not run on MPS; the Mac script uses 3 clients per dataset, which avoids them.
- **Group label spaces.** A PACFL group must not mix label spaces. The server stops with an error if it does;
  lower `--pacfl_cluster_alpha` or set `rt_group_by: dataset`.
- `docs/rt_attention_revision.md` refers to log folders that are not part of this repository.
