#!/bin/bash
# New-client comparison: PSI (RT before training) vs baseline mappings. Originals untouched.
# Usage: bash run_newclient_psi_new.sh [START_ROUND=25] [CUDA=cuda:0] [NEW_CLIENT_EPOCHS=30]
set -e
ROUND=${1:-25}
CUDA=${2:-cuda:0}
EP=${3:-30}
SEED=15698
ALG=GeFL_gan_pacfl_iid
COMMON="--seed=$SEED --algorithm=$ALG --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
        --num_new_clients=1 --device=$CUDA --pacfl_cluster_alpha=20 --pacfl_basis_budget=20 \
        --start_mapping_epoch=$ROUND --exp_conf=./configs/het-iid-exp_rt_filter_new.yaml"

NAMES=(psi_filter image_bi missing_link feature_bi)
declare -A LM=([psi_filter]=rt [image_bi]=image-bi [missing_link]=missing_link [feature_bi]=feature-bi)
declare -A LABEL=([psi_filter]="PSI (filter)" [image_bi]="Ours (image-bi)" [missing_link]="Missing Link" [feature_bi]="feature-bi")

# 1. FL runs (all through server_new.py -> same test metric + server_checkpoints.pth / server_global_model.pth)
for N in "${NAMES[@]}"; do
  D=logs/nc${ROUND}_${N}/$ALG
  [ -f $D/server_checkpoints.pth ] && { echo "skip FL $N (done)"; continue; }
  python main_new.py $COMMON --label_mapping=${LM[$N]} --exp_timestamp=nc${ROUND}_${N}
done

# 2. new client: fine-tune each global model; scratch baseline once
for MODE in single super; do
  for N in "${NAMES[@]}"; do
    python new_client_run_new.py --mode finetune --model_path=logs/nc${ROUND}_${N}/$ALG/server_checkpoints.pth \
      --device=$CUDA --dataset_mode=$MODE --new_client_epochs=$EP --exp_timestamp=nc
  done
  python new_client_run_new.py --mode scratch --model_path=logs/nc${ROUND}_psi_filter/$ALG/server_checkpoints.pth \
    --device=$CUDA --dataset_mode=$MODE --new_client_epochs=$EP --exp_timestamp=nc
done

# 3. comparison plots (one figure per dataset; super = SuperDataset)
for MODE in single super; do
  CSVS=()
  for N in "${NAMES[@]}"; do
    CSVS+=("${LABEL[$N]}:logs/nc${ROUND}_${N}/$ALG/new_client_nc/our_finetune_global_${MODE}_dataset")
  done
  CSVS+=("scratch:logs/nc${ROUND}_psi_filter/$ALG/new_client_nc/baseline_${MODE}_dataset")
  python utils/plot_comparison_new.py --csvs "${CSVS[@]}" --output_dir plot/new_client_psi/${MODE}
done
