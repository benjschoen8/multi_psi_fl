#!/bin/bash
# Global-model accuracy: PSI vs baseline mappings, small scale on Mac GPU (mps). Originals untouched.
# Usage: bash run_mac_global_acc_new.sh [ROUNDS=10] [START=5] [CLIENTS_PER_DATASET=3]
# START = round the BASELINES start mapping + global model; PSI always starts at round 1.
set -e
ROUNDS=${1:-10}
START=${2:-5}
NC=${3:-3}
export PYTORCH_ENABLE_MPS_FALLBACK=1
TAG=mac_r${ROUNDS}_s${START}_c${NC}

# same yaml for every method; only rounds changed (rt_* keys are ignored by baselines)
sed "s/^global_rounds: .*/global_rounds: ${ROUNDS}/" configs/het-iid-exp_rt_filter_mps_new.yaml > configs/het-iid-exp_${TAG}_new.yaml

COMMON="--seed=15698 --algorithm=GeFL_gan_pacfl_iid \
  --num_train_mnist=$NC --num_train_emnist=$NC --num_train_cifar10=$NC \
  --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --num_new_clients=1 \
  --device=mps --pacfl_cluster_alpha=20 --pacfl_basis_budget=20  \
  --exp_conf=./configs/het-iid-exp_${TAG}_new.yaml"

for LM in rt image-bi missing_link feature-bi image-cs; do
  # PSI: mapping exists before round 1 -> global model trained from round 1.
  # baselines: mapping needs trained generators -> global model from round START.
  S=$START; [ "$LM" = rt ] && S=1
  D=logs/${TAG}_${LM}/GeFL_gan_pacfl_iid
  [ -f $D/global_model_acc_mix.csv ] && [ "$(wc -l < $D/global_model_acc_mix.csv)" -gt $((ROUNDS - S + 1)) ] \
    && { echo "skip $LM (done)"; continue; }
  python3 main_new.py $COMMON --start_mapping_epoch=$S --label_mapping=$LM --exp_timestamp=${TAG}_${LM}
done

# PSI, attention versions (client-written multilingual descriptions, en/zh/es):
#   attn_filter = description attention + image filter ; attn = description attention only
for V in attn_filter attn; do
  sed "s/^global_rounds: .*/global_rounds: ${ROUNDS}/" configs/het-iid-exp_rt_${V}_mps_new.yaml > configs/het-iid-exp_${TAG}_${V}_new.yaml
  D=logs/${TAG}_rt_${V}/GeFL_gan_pacfl_iid
  if [ -f $D/global_model_acc_mix.csv ] && [ "$(wc -l < $D/global_model_acc_mix.csv)" -gt $ROUNDS ]; then
    echo "skip rt_${V} (done)"
  else
    python3 main_new.py ${COMMON/het-iid-exp_${TAG}_new.yaml/het-iid-exp_${TAG}_${V}_new.yaml} --start_mapping_epoch=1 \
      --label_mapping=rt --exp_timestamp=${TAG}_rt_${V}
  fi
done

python3 plot/plot_global_model_acc_new.py \
  --psi "PSI (filter)=logs/${TAG}_rt/GeFL_gan_pacfl_iid" \
  --psi "PSI (attn_filter)=logs/${TAG}_rt_attn_filter/GeFL_gan_pacfl_iid" \
  --psi "PSI (attn)=logs/${TAG}_rt_attn/GeFL_gan_pacfl_iid" \
  --run "Ours (image-bi)=logs/${TAG}_image-bi/GeFL_gan_pacfl_iid" \
  --run "Missing Link=logs/${TAG}_missing_link/GeFL_gan_pacfl_iid" \
  --run "feature-bi=logs/${TAG}_feature-bi/GeFL_gan_pacfl_iid" \
  --run "cosine-similarity=logs/${TAG}_image-cs/GeFL_gan_pacfl_iid" \
  --datasets MNIST EMNIST CIFAR10 --out plot/global_accuracy_plots/${TAG}_psi
echo "plots: plot/global_accuracy_plots/${TAG}_psi/Global_Acc_{MNIST,EMNIST,CIFAR10,mix}.pdf/.png"
