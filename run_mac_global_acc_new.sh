#!/bin/bash
# Global-model accuracy: PSI vs baseline mappings. Defaults = original paper settings
# (configs/het-iid-exp.yaml + main.py): 45 rounds, 10 clients per dataset, all samples,
# baselines build their mapping and start the global model at round 25.
# Usage: bash run_mac_global_acc_new.sh [ROUNDS=45] [START=25] [CLIENTS_PER_DATASET=10] [DEVICE=auto] [WARMUP=START-1] [CAP=0] [METHOD=all]
#   START  = round the BASELINES build their mapping + start the global model (original: 25)
#   WARMUP = generator-only rounds before the PSI global models start (default START-1: PSI starts at
#            START, same round as the baselines; PSI tables still exist before round 1)
#   CAP    = max training samples per client (0 = all, original)
#   DEVICE = auto (cuda:0 > mps > cpu) | cpu | mps | cuda:N
#   METHOD = all (default) | one method | comma list. Names = log-dir suffixes:
#            baselines: image-bi missing_link feature-bi image-cs
#            PSI:       rt (FPSI-DescFilter) rt_attn_filter rt_attn rt_cpsi_helper rt_cpsi_2pc rt_cpsi_tag rt_psi_tag_hash rt_psi_trivial
#            The plot still includes any other methods already finished under the same TAG.
# Mac-sized example: bash run_mac_global_acc_new.sh 25 15 3 mps "" 2000
# One method only:   bash run_mac_global_acc_new.sh 25 15 3 mps "" 2000 rt_cpsi_tag
set -e
ROUNDS=${1:-45}
START=${2:-25}
NC=${3:-10}
DEVICE=${4:-auto}
WARMUP=${5:-$((START - 1))}
CAP=${6:-0}
METHOD=${7:-all}
# positional args are easy to shift: fail fast instead of deep inside torch
for V in ROUNDS START NC CAP; do [[ ${!V} =~ ^[0-9]+$ ]] || { echo "$V must be an integer, got '${!V}' (args: ROUNDS START NC DEVICE WARMUP CAP METHOD)"; exit 1; }; done
[[ $DEVICE =~ ^(auto|cpu|mps|cuda(:[0-9]+)?)$ ]] || { echo "DEVICE must be auto|cpu|mps|cuda:N, got '$DEVICE' (args: ROUNDS START NC DEVICE WARMUP CAP METHOD)"; exit 1; }
[ "$NC" -le 100 ] || { echo "NC=$NC clients per dataset looks wrong (did CAP land in slot 3?)"; exit 1; }
ALL_METHODS="rt image-bi missing_link feature-bi image-cs rt_attn_filter rt_attn rt_cpsi_helper rt_cpsi_2pc rt_cpsi_tag rt_psi_tag_hash rt_psi_trivial"
if [ "$METHOD" != all ]; then
  for M in ${METHOD//,/ }; do
    [[ " $ALL_METHODS " == *" $M "* ]] || { echo "unknown METHOD '$M'; choose from: all $ALL_METHODS"; exit 1; }
  done
fi
want() { [ "$METHOD" = all ] || [[ ",$METHOD," == *",$1,"* ]]; }
echo "methods: $METHOD"
[ "$WARMUP" -lt "$START" ] || { echo "WARMUP ($WARMUP) must be < START ($START)"; exit 1; }
PSI_START=$((WARMUP + 1))
if [ "$DEVICE" = auto ]; then
  DEVICE=$(python3 -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')")
fi
echo "device: $DEVICE"
export PYTORCH_ENABLE_MPS_FALLBACK=1        # only used on mps
TAG=r${ROUNDS}_s${START}_w${WARMUP}_c${NC}_cap${CAP}

# same yaml for every method; only rounds + sample cap changed (rt_* keys are ignored by baselines)
cfg() { sed -e "s/^global_rounds: .*/global_rounds: ${ROUNDS}/" -e "s/^max_client_samples: .*/max_client_samples: ${CAP}/" "$1" > "$2"; }
cfg configs/het-iid-exp_rt_filter_mps_new.yaml configs/het-iid-exp_${TAG}_new.yaml

COMMON="--seed=15698 --algorithm=GeFL_gan_pacfl_iid \
  --num_train_mnist=$NC --num_train_emnist=$NC --num_train_cifar10=$NC \
  --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --num_new_clients=1 \
  --device=$DEVICE --pacfl_cluster_alpha=20 --pacfl_basis_budget=20  \
  --exp_conf=./configs/het-iid-exp_${TAG}_new.yaml"

for LM in rt image-bi missing_link feature-bi image-cs; do
  # PSI: table built before round 1; global model after the generator warm-up (WARMUP+1).
  # baselines: mapping needs trained models/generators -> built at START, global model from START.
  want $LM || continue
  S=$START; [ "$LM" = rt ] && S=$PSI_START
  D=logs/${TAG}_${LM}/GeFL_gan_pacfl_iid
  [ -f $D/global_model_acc_mix.csv ] && [ "$(wc -l < $D/global_model_acc_mix.csv)" -gt $((ROUNDS - S + 1)) ] \
    && { echo "skip $LM (done)"; continue; }
  python3 main_new.py $COMMON --start_mapping_epoch=$S --label_mapping=$LM --exp_timestamp=${TAG}_${LM}
done

# PSI, attention versions (client-written multilingual descriptions, en/zh/es):
#   attn_filter = description attention + image filter ; attn = description attention only
#   cpsi_* = attn_filter (description + image in one PSI message) as server-free circuit-PSI:
#     cpsi_helper = BFV PSI + 2PC with a helper client      (server sees final table only)
#     cpsi_2pc    = VOLE PSI + per-pair 2PC + 2PC grouping  (server sees final table only, no helper)
#     cpsi_tag    = VOLE PSI + per-pair 2PC -> equality tags (server sees pairwise matches, groups itself)
#   psi_tag_hash  = per-check fuzzy-PSI tags, each side sends hash(tag_text|tag_image|tag_verify);
#                   server: AND via equal hashes, Mutual, grouping. No circuit, no rival margin.
#   psi_trivial   = every client describes a class by the same agreed name (3, A, car); exact DH-PSI
#                   on the names gives the relation table directly (sanity / upper-bound baseline).
for V in attn_filter attn cpsi_helper cpsi_2pc cpsi_tag psi_tag_hash psi_trivial; do
  want rt_${V} || continue
  cfg configs/het-iid-exp_rt_${V}_mps_new.yaml configs/het-iid-exp_${TAG}_${V}_new.yaml
  D=logs/${TAG}_rt_${V}/GeFL_gan_pacfl_iid
  if [ -f $D/global_model_acc_mix.csv ] && [ "$(wc -l < $D/global_model_acc_mix.csv)" -gt $((ROUNDS - PSI_START + 1)) ]; then
    echo "skip rt_${V} (done)"
  else
    python3 main_new.py ${COMMON/het-iid-exp_${TAG}_new.yaml/het-iid-exp_${TAG}_${V}_new.yaml} --start_mapping_epoch=$PSI_START \
      --label_mapping=rt --exp_timestamp=${TAG}_rt_${V}
  fi
done

python3 plot/plot_global_model_acc_new.py \
  --psi "FPSI-DescFilter [CKKS]=logs/${TAG}_rt/GeFL_gan_pacfl_iid" \
  --psi "FPSI-AttnImage [CKKS]=logs/${TAG}_rt_attn_filter/GeFL_gan_pacfl_iid" \
  --psi "FPSI-AttnText [CKKS]=logs/${TAG}_rt_attn/GeFL_gan_pacfl_iid" \
  --psi "CPSI-Helper [BFV]=logs/${TAG}_rt_cpsi_helper/GeFL_gan_pacfl_iid" \
  --psi "CPSI-2PC [VOLE]=logs/${TAG}_rt_cpsi_2pc/GeFL_gan_pacfl_iid" \
  --psi "CPSI-Tag [VOLE]=logs/${TAG}_rt_cpsi_tag/GeFL_gan_pacfl_iid" \
  --psi "PSI-TagHash [OPPRF]=logs/${TAG}_rt_psi_tag_hash/GeFL_gan_pacfl_iid" \
  --psi "PSI-Trivial [DH]=logs/${TAG}_rt_psi_trivial/GeFL_gan_pacfl_iid" \
  --run "Ours (image-bi)=logs/${TAG}_image-bi/GeFL_gan_pacfl_iid" \
  --run "Missing Link=logs/${TAG}_missing_link/GeFL_gan_pacfl_iid" \
  --run "feature-bi=logs/${TAG}_feature-bi/GeFL_gan_pacfl_iid" \
  --run "cosine-similarity=logs/${TAG}_image-cs/GeFL_gan_pacfl_iid" \
  --datasets MNIST EMNIST CIFAR10 --out plot/global_accuracy_plots/${TAG}_psi
echo "plots: plot/global_accuracy_plots/${TAG}_psi/Global_Acc_{MNIST,EMNIST,CIFAR10,mix}.pdf/.png"
