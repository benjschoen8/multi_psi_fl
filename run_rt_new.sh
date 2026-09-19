#!/bin/bash
# RT protocol (PSI before training) runs + comparison plots. Originals untouched.
# Usage: bash run_rt_new.sh [START_ROUND] [CUDA]
set -e
ROUND=${1:-25}
CUDA=${2:-cuda:0}
SEED=15698
COMMON="--seed=$SEED --algorithm=GeFL_gan_pacfl_iid --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
        --device=$CUDA --pacfl_cluster_alpha=20 --pacfl_basis_budget=20 --start_mapping_epoch=$ROUND --label_mapping=rt"

# attnfilter = client-written multilingual descriptions -> attention over public anchor words + image filter
# attntext   = the same description attention only, no images
# 1. training runs (RT once before round 1, then generator-only FL)
for M in filter affscan attnfilter attntext; do
  python main_new.py $COMMON --exp_conf=./configs/het-iid-exp_rt_${M}_new.yaml --exp_timestamp=start${ROUND}_noniid_gan_psi_${M}
done

# 2. diagnostics per run  -> logs/.../rt_diag_plots/
for M in filter affscan attnfilter attntext; do
  python plot/plot_rt_diagnostics_new.py logs/start${ROUND}_noniid_gan_psi_${M}/GeFL_gan_pacfl_iid
done

# 3. global model accuracy: baselines (existing logs) + PSI
python plot/plot_global_model_acc_new.py \
  --run "Ours=logs/start${ROUND}_noniid_gan_ours/GeFL_gan_pacfl_iid" \
  --run "Missing Link=logs/start${ROUND}_noniid_gan_missinglink/GeFL_gan_pacfl_iid" \
  --run "feature=logs/start${ROUND}_noniid_gan_feature/GeFL_gan_pacfl_iid" \
  --run "cosine-similarity=logs/start${ROUND}_noniid_gan_cs/GeFL_gan_pacfl_iid" \
  --psi "PSI (filter)=logs/start${ROUND}_noniid_gan_psi_filter/GeFL_gan_pacfl_iid" \
  --psi "PSI (affscan)=logs/start${ROUND}_noniid_gan_psi_affscan/GeFL_gan_pacfl_iid" \
  --psi "PSI (attn_filter)=logs/start${ROUND}_noniid_gan_psi_attnfilter/GeFL_gan_pacfl_iid" \
  --psi "PSI (attn)=logs/start${ROUND}_noniid_gan_psi_attntext/GeFL_gan_pacfl_iid" \
  --datasets MNIST EMNIST CIFAR10 --out plot/global_accuracy_plots/global_${ROUND}_noniid_psi

# 4. mapping metrics: baselines (existing offline csvs) + PSI as flat lines
D=label_mapping/pacfl_3cluster/noniid
python plot/plot_label_mapping_method_acc_new.py \
  --method "Missing Link=$D/missing_link_seed${SEED}/label_mapping/offline_missing_link_noniid_mapping_acc.csv::missing_threshold" \
  --method "Improve=$D/improve_single_seed${SEED}/label_mapping/offline_improve_single_noniid_mapping_acc.csv" \
  --method "Feature=$D/feature_seed${SEED}/label_mapping/offline_feature_noniid_mapping_acc.csv" \
  --method "Improve_noniid=$D/improve_single_noniid_seed${SEED}/label_mapping/offline_improve_single_noniid_noniid_mapping_acc.csv" \
  --psi "PSI (filter)=logs/start${ROUND}_noniid_gan_psi_filter/GeFL_gan_pacfl_iid/rt_mapping_acc.csv" \
  --psi "PSI (affscan)=logs/start${ROUND}_noniid_gan_psi_affscan/GeFL_gan_pacfl_iid/rt_mapping_acc.csv" \
  --psi "PSI (attn_filter)=logs/start${ROUND}_noniid_gan_psi_attnfilter/GeFL_gan_pacfl_iid/rt_mapping_acc.csv" \
  --psi "PSI (attn)=logs/start${ROUND}_noniid_gan_psi_attntext/GeFL_gan_pacfl_iid/rt_mapping_acc.csv" \
  --round $ROUND --out $D/psi_compare_seed${SEED}
