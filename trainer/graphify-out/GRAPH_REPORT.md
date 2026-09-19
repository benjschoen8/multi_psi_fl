# Graph Report - trainer  (2026-09-18)

## Corpus Check
- Corpus is ~41,282 words - fits in a single context window. You may not need a graph.

## Summary
- 766 nodes · 1825 edges · 31 communities (30 shown, 1 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Shared Imports & Server Hub
- Local Baseline Server
- BaseFL Client Family
- FedTED DDPM v2
- FedTED DDPM
- GeFL DDPM Baseline
- GeFL DeepInversion Gen
- GeFL GAN PACFL IID
- GeFL DDPM Total
- GeFL GAN+DDPM
- GeFL Total GAN
- Total GAN Non-IID
- Total GAN Separate
- GeFL GAN Missing Link
- GeFL Slamdunk
- FedTED Dirichlet
- GeFL DDPM
- GeFL DeepInversion
- GeFL GAN Baseline
- GeFL Core
- FedTED
- GeFL DDIM Baseline
- DDPM Total Public
- DDPM Total Separate
- Label Mapping Results
- Ours GeFL
- DDPM Total DeepInversion
- New Client Trainer
- BaseFL DDPM
- GeFL Local
- Feature Distillation Server

## God Nodes (most connected - your core abstractions)
1. `Server` - 50 edges
2. `Client` - 42 edges
3. `get_mapping()` - 17 edges
4. `Server` - 14 edges
5. `Server` - 14 edges
6. `Server` - 13 edges
7. `Server` - 13 edges
8. `Client` - 12 edges
9. `Server` - 12 edges
10. `Server` - 12 edges

## Surprising Connections (you probably didn't know these)
- `Server` --inherits--> `Node`  [EXTRACTED]
  ../mnt/multi-FL-main/trainer/BaseFL/server.py → ../mnt/multi-FL-main/trainer/BaseFL/client.py
- `Client` --inherits--> `Client`  [EXTRACTED]
  ../mnt/multi-FL-main/trainer/BaseFL_DDPM/client.py → ../mnt/multi-FL-main/trainer/BaseFL/client.py
- `Client` --inherits--> `Client`  [EXTRACTED]
  ../mnt/multi-FL-main/trainer/FedTED/client.py → ../mnt/multi-FL-main/trainer/BaseFL/client.py
- `Client` --inherits--> `Client`  [EXTRACTED]
  ../mnt/multi-FL-main/trainer/FedTED_DDPM_2/client.py → ../mnt/multi-FL-main/trainer/BaseFL/client.py
- `Client` --inherits--> `Client`  [EXTRACTED]
  ../mnt/multi-FL-main/trainer/FedTED_DDPM/client.py → ../mnt/multi-FL-main/trainer/BaseFL/client.py

## Import Cycles
- None detected.

## Communities (31 total, 1 thin omitted)

### Community 0 - "Shared Imports & Server Hub"
Cohesion: 0.13
Nodes (48): collections, contextlib, copy, csv, json, label_mapping_label_mapping_utils, label_mapping_slam_dunk, math (+40 more)

### Community 1 - "Local Baseline Server"
Cohesion: 0.06
Nodes (9): reconstruct feature extractor to get a generic model, Server, n_k_and_weights: [..., (n_k, w_k), ....], where n_k is the number of samples…, Select some fraction of all clients., Server, server don't distribute model anymore, Server, Server (+1 more)

### Community 2 - "BaseFL Client Family"
Cohesion: 0.08
Nodes (14): Client, Node, train node's model by local train dataset, evaluate node's model by local test dataset :return correct, test_loss, A computation node, could be clients, servers or any computed devices. It can…, Client, Here, client is only for test personal performance, Client (+6 more)

### Community 3 - "FedTED DDPM v2"
Cohesion: 0.11
Nodes (7): Client, FedTED with communication efficient strategy, Train ConditionalGenerator in feature space (FedTED)., Train global inference model using DDPM synthetic data with FedTED MSE+CE loss., Generate prototype embeddings for each global class via ConditionalGenerator., FedAvg aggregation for DDPM., Server

### Community 4 - "FedTED DDPM"
Cohesion: 0.12
Nodes (5): Client, FedTED with communication efficient strategy, reconstruct feature extractor to get a generic model, FedAvg aggregation for Generator, Server

### Community 5 - "GeFL DDPM Baseline"
Cohesion: 0.12
Nodes (5): Client, Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server

### Community 6 - "GeFL DeepInversion Gen"
Cohesion: 0.12
Nodes (8): Client, Client for GeFL (DeepInversion / Data-Free KD), Overrides BaseFL Node update() method. Local update: 1) Train local target…, Train local heterogeneous target model using real data., Save a grid image for each global_id to inspect generation quality., Server, format_class_name(), safe_text()

### Community 7 - "GeFL GAN PACFL IID"
Cohesion: 0.12
Nodes (6): Client, Local DCGAN training (train G and D)., Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server

### Community 8 - "GeFL DDPM Total"
Cohesion: 0.12
Nodes (7): Client, GeFL_DDPM_baseline_total Client, Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server, utils_ddpm_nets

### Community 9 - "GeFL GAN+DDPM"
Cohesion: 0.12
Nodes (6): Client, Local DCGAN training (train G and D)., Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server

### Community 10 - "GeFL Total GAN"
Cohesion: 0.13
Nodes (6): Client, Local DCGAN training (train G and D)., Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server

### Community 11 - "Total GAN Non-IID"
Cohesion: 0.12
Nodes (6): Client, Local DCGAN training (train G and D)., Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server

### Community 12 - "Total GAN Separate"
Cohesion: 0.13
Nodes (6): Client, Local DCGAN training (train G and D)., Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server

### Community 13 - "GeFL GAN Missing Link"
Cohesion: 0.12
Nodes (6): Client, Local DCGAN training (train G and D)., Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server

### Community 14 - "GeFL Slamdunk"
Cohesion: 0.13
Nodes (6): Client, Local DCGAN training (train G and D)., Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server

### Community 15 - "FedTED Dirichlet"
Cohesion: 0.14
Nodes (4): Client, FedTED with communication efficient strategy, reconstruct feature extractor to get a generic model, Server

### Community 16 - "GeFL DDPM"
Cohesion: 0.14
Nodes (5): Client, Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server

### Community 17 - "GeFL DeepInversion"
Cohesion: 0.14
Nodes (6): Client, Client for GeFL (DeepInversion / Data-Free KD), Overrides BaseFL Node update() method. Local update: 1) Train local target…, Train local heterogeneous target model using real data., Save a grid image for each global_id to inspect generation quality., Server

### Community 18 - "GeFL GAN Baseline"
Cohesion: 0.13
Nodes (6): Client, Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., Local DCGAN training (train G and D)., FedAvg aggregation for Generator, Server

### Community 19 - "GeFL Core"
Cohesion: 0.13
Nodes (6): Client, Local DCGAN training (train G and D)., Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server

### Community 20 - "FedTED"
Cohesion: 0.14
Nodes (4): Client, FedTED with communication efficient strategy, reconstruct feature extractor to get a generic model, Server

### Community 21 - "GeFL DDIM Baseline"
Cohesion: 0.15
Nodes (5): Client, Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server

### Community 22 - "DDPM Total Public"
Cohesion: 0.15
Nodes (5): Client, Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server

### Community 23 - "DDPM Total Separate"
Cohesion: 0.14
Nodes (5): Client, Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server

### Community 24 - "Label Mapping Results"
Cohesion: 0.28
Nodes (19): get_class_name_mapping(), get_cs_mapping_25(), get_feature_mapping_25(), get_mapping(), get_ours_10(), get_ours_15(), get_ours_20(), get_ours_25() (+11 more)

### Community 25 - "Ours GeFL"
Cohesion: 0.14
Nodes (6): Client, Local DCGAN training (train G and D)., Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., FedAvg aggregation for Generator, Server

### Community 26 - "DDPM Total DeepInversion"
Cohesion: 0.15
Nodes (5): Client, Client for GeFL (DeepInversion / Data-Free KD), Overrides BaseFL Node update() method. Local update: 1) Train local target…, Train local heterogeneous target model using real data., Server

### Community 27 - "New Client Trainer"
Cohesion: 0.14
Nodes (9): argparse, data_datasets, datetime, importlib, BaseNewClientTrainer, time, utils_csv_logger, utils_logger (+1 more)

### Community 28 - "BaseFL DDPM"
Cohesion: 0.15
Nodes (4): Client, Here, client is only for test personal performance, FedAvg aggregation for Generator, Server

### Community 29 - "GeFL Local"
Cohesion: 0.14
Nodes (5): Client, Local DCGAN training (train G and D)., Overrides BaseFL Node update() method. GeFL local update: 1) Train local DCGAN…, Train local heterogeneous target model using real + generated data., Server

## Knowledge Gaps
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Server` connect `Local Baseline Server` to `Shared Imports & Server Hub`, `BaseFL Client Family`, `FedTED DDPM v2`, `FedTED DDPM`, `GeFL DDPM Baseline`, `GeFL DeepInversion Gen`, `GeFL GAN PACFL IID`, `GeFL DDPM Total`, `GeFL GAN+DDPM`, `GeFL Total GAN`, `Total GAN Non-IID`, `Total GAN Separate`, `GeFL GAN Missing Link`, `GeFL Slamdunk`, `FedTED Dirichlet`, `GeFL DDPM`, `GeFL DeepInversion`, `GeFL GAN Baseline`, `GeFL Core`, `FedTED`, `GeFL DDIM Baseline`, `DDPM Total Public`, `DDPM Total Separate`, `Ours GeFL`, `DDPM Total DeepInversion`, `BaseFL DDPM`, `GeFL Local`, `Feature Distillation Server`?**
  _High betweenness centrality (0.351) - this node is a cross-community bridge._
- **Why does `Client` connect `BaseFL Client Family` to `Shared Imports & Server Hub`, `FedTED DDPM v2`, `FedTED DDPM`, `GeFL DDPM Baseline`, `GeFL DeepInversion Gen`, `GeFL GAN PACFL IID`, `GeFL DDPM Total`, `GeFL GAN+DDPM`, `GeFL Total GAN`, `Total GAN Non-IID`, `Total GAN Separate`, `GeFL GAN Missing Link`, `GeFL Slamdunk`, `FedTED Dirichlet`, `GeFL DDPM`, `GeFL DeepInversion`, `GeFL GAN Baseline`, `GeFL Core`, `FedTED`, `GeFL DDIM Baseline`, `DDPM Total Public`, `DDPM Total Separate`, `Ours GeFL`, `DDPM Total DeepInversion`, `BaseFL DDPM`, `GeFL Local`?**
  _High betweenness centrality (0.266) - this node is a cross-community bridge._
- **Why does `Node` connect `BaseFL Client Family` to `Shared Imports & Server Hub`, `Local Baseline Server`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Should `Shared Imports & Server Hub` be split into smaller, more focused modules?**
  _Cohesion score 0.1292517006802721 - nodes in this community are weakly interconnected._
- **Should `Local Baseline Server` be split into smaller, more focused modules?**
  _Cohesion score 0.06183574879227053 - nodes in this community are weakly interconnected._
- **Should `BaseFL Client Family` be split into smaller, more focused modules?**
  _Cohesion score 0.07807807807807808 - nodes in this community are weakly interconnected._
- **Should `FedTED DDPM v2` be split into smaller, more focused modules?**
  _Cohesion score 0.11333333333333333 - nodes in this community are weakly interconnected._