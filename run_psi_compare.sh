#!/bin/bash
# One-seed comparison of the global inference model under five label mappings:
#   class_name  = oracle: class names shared in the clear (no privacy)
#   image-bi    = original generator-based bi-direction mapping
#   feature-bi  = bi-direction mapping on averaged client features
#   image-cs    = cosine similarity of generated images
#   psi_trivial = same names as class_name, matched privately by exact DH-PSI (label_mapping/psi_trivial_new.py)
# usage: bash run_psi_compare.sh [SEED] [DEVICE]        e.g. bash run_psi_compare.sh 15698 mps
SEED=${1:-15698}
DEVICE=${2:-mps}
START=25   # same mapping round for all five, so only the mapping differs

run () {   # $1 = label_mapping, $2 = log-folder tag
    [ -f ./logs/seed${SEED}_iid_$2/GeFL_gan_pacfl_iid/server_checkpoints_45.pth ] && { echo "skip $1 (already finished)"; return; }
    python3 main.py --seed=$SEED --algorithm=GeFL_gan_pacfl_iid --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
        --device=$DEVICE --exp_conf=./configs/het-iid-exp.yaml --pacfl_cluster_alpha=10 --pacfl_basis_budget=20 \
        --label_mapping=$1 --start_mapping_epoch=$START --exp_timestamp=seed${SEED}_iid_$2
}

run class_name  class_name
run image-bi    gan_weight     # same folder exp_run.sh trains into, so a finished run there is reused
run feature-bi  feature_bi
run image-cs    image_cs
run psi_trivial psi_trivial

python3 plot/plot_global_acc_psi.py --seed=$SEED
