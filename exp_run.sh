#!/bin/bash

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Experiment Timestamp: $TIMESTAMP"

TOTAL_START=$SECONDS
CUDA="cuda:1"
DEVICE="mps"   # offline mapping device: mps on Mac, cuda:1 on the GPU server


# ------------------------------------------------------------------------
# Noniid
# ------------------------------------------------------------------------
# python main.py --seed=1 --algorithm=GeFL_gan_pacfl_iid --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
#             --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml --pacfl_cluster_alpha=22 --pacfl_basis_budget=20

# python main.py --seed=42 --algorithm=GeFL_gan_pacfl_iid --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
#             --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml --pacfl_cluster_alpha=28 --pacfl_basis_budget=15

# python main.py --seed=758 --algorithm=GeFL_gan_pacfl_iid --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
#             --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml --pacfl_cluster_alpha=22 --pacfl_basis_budget=20

# python main.py --seed=1248 --algorithm=GeFL_gan_pacfl_iid --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
#             --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml --pacfl_cluster_alpha=25 --pacfl_basis_budget=10

# python main.py --seed=15698 --algorithm=GeFL_gan_pacfl_iid --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
#             --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml --pacfl_cluster_alpha=20 --pacfl_basis_budget=20

# python main.py --seed=15698 --algorithm=GeFL_gan_pacfl_iid --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
#             --device=cuda:0 --exp_conf=./configs/het-iid-exp.yaml --pacfl_cluster_alpha=10 --pacfl_basis_budget=20

# ------------------------------------------------------------------------
# iid
# ------------------------------------------------------------------------
# python main.py --seed=1 --algorithm=GeFL_gan_pacfl_iid --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
#             --device=cuda:0 --exp_conf=./configs/het-iid-exp.yaml --pacfl_cluster_alpha=10 --pacfl_basis_budget=20

# python main.py --seed=42 --algorithm=GeFL_gan_pacfl_iid --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
#             --device=cuda:0 --exp_conf=./configs/het-iid-exp.yaml --pacfl_cluster_alpha=10 --pacfl_basis_budget=20

# python main.py --seed=758 --algorithm=GeFL_gan_pacfl_iid --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
#             --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml --pacfl_cluster_alpha=10 --pacfl_basis_budget=20

# python main.py --seed=1248 --algorithm=GeFL_gan_pacfl_iid --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
#             --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml --pacfl_cluster_alpha=10 --pacfl_basis_budget=20

# python main.py --seed=15698 --algorithm=GeFL_gan_pacfl_iid --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
#             --device=cuda:0 --exp_conf=./configs/het-iid-exp.yaml --pacfl_cluster_alpha=10 --pacfl_basis_budget=20


# ------------------------------------------------------------------------
# Noniid
# ------------------------------------------------------------------------

# python label_mapping/offline_mapping_noniid_global.py --seed=1 --label_mapping=improve_single --log_dir=./logs/seed1_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=42 --label_mapping=improve_single --log_dir=./logs/seed42_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=758 --label_mapping=improve_single --log_dir=./logs/seed758_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=1248 --label_mapping=improve_single --log_dir=./logs/seed1248_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=15698 --label_mapping=improve_single --log_dir=./logs/seed15698_noniid_gan_weight/GeFL_gan_pacfl_iid


# python label_mapping/offline_mapping_noniid_global.py --seed=1 --label_mapping=improve_single_noniid --log_dir=./logs/seed1_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=42 --label_mapping=improve_single_noniid --log_dir=./logs/seed42_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=758 --label_mapping=improve_single_noniid --log_dir=./logs/seed758_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=1248 --label_mapping=improve_single_noniid --log_dir=./logs/seed1248_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=15698 --label_mapping=improve_single_noniid --log_dir=./logs/seed15698_noniid_gan_weight/GeFL_gan_pacfl_iid


# python label_mapping/offline_mapping_noniid_global.py --seed=1 --label_mapping=missing_link --log_dir=./logs/seed1_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=42 --label_mapping=missing_link --log_dir=./logs/seed42_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=758 --label_mapping=missing_link --log_dir=./logs/seed758_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=1248 --label_mapping=missing_link --log_dir=./logs/seed1248_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=15698 --label_mapping=missing_link --log_dir=./logs/seed15698_noniid_gan_weight/GeFL_gan_pacfl_iid

# python label_mapping/offline_mapping_noniid_global.py --seed=1 --label_mapping=missing_link_single --log_dir=./logs/seed1_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=42 --label_mapping=missing_link_single --log_dir=./logs/seed42_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=758 --label_mapping=missing_link_single --log_dir=./logs/seed758_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=1248 --label_mapping=missing_link_single --log_dir=./logs/seed1248_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=15698 --label_mapping=missing_link_single --log_dir=./logs/seed15698_noniid_gan_weight/GeFL_gan_pacfl_iid

# python label_mapping/offline_mapping_noniid_global.py --seed=1 --label_mapping=psi_trivial --log_dir=./logs/seed1_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=42 --label_mapping=psi_trivial --log_dir=./logs/seed42_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=758 --label_mapping=psi_trivial --log_dir=./logs/seed758_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=1248 --label_mapping=psi_trivial --log_dir=./logs/seed1248_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=15698 --label_mapping=psi_trivial --log_dir=./logs/seed15698_noniid_gan_weight/GeFL_gan_pacfl_iid



# python label_mapping/offline_mapping_noniid_global.py --seed=1 --label_mapping=image-cs --log_dir=./logs/seed1_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=42 --label_mapping=image-cs --log_dir=./logs/seed42_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=758 --label_mapping=image-cs --log_dir=./logs/seed758_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=1248 --label_mapping=image-cs --log_dir=./logs/seed1248_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=15698 --label_mapping=image-cs --log_dir=./logs/seed15698_noniid_gan_weight/GeFL_gan_pacfl_iid


# python label_mapping/offline_mapping_noniid_global.py --seed=1 --label_mapping=feature --log_dir=./logs/seed1_noniid_gan_weight/GeFL_gan_pacfl_iid \
#     --mnist_split ./data/raw/splits/MNIST_C10_New0_alpha0p1_seed1.json \
#     --emnist_split ./data/raw/splits/EMNIST_C10_New0_alpha0p1_seed1.json \
#     --cifar10_split ./data/raw/splits/CIFAR10_C10_New0_alpha0p1_seed1.json

# python label_mapping/offline_mapping_noniid_global.py --seed=42 --label_mapping=feature --log_dir=./logs/seed42_noniid_gan_weight/GeFL_gan_pacfl_iid \
#     --mnist_split ./data/raw/splits/MNIST_C10_New0_alpha0p1_seed42.json \
#     --emnist_split ./data/raw/splits/EMNIST_C10_New0_alpha0p1_seed42.json \
#     --cifar10_split ./data/raw/splits/CIFAR10_C10_New0_alpha0p1_seed42.json

# python label_mapping/offline_mapping_noniid_global.py --seed=758 --label_mapping=feature --log_dir=./logs/seed758_noniid_gan_weight/GeFL_gan_pacfl_iid \
#     --mnist_split ./data/raw/splits/MNIST_C10_New0_alpha0p1_seed758.json \
#     --emnist_split ./data/raw/splits/EMNIST_C10_New0_alpha0p1_seed758.json \
#     --cifar10_split ./data/raw/splits/CIFAR10_C10_New0_alpha0p1_seed758.json

# python label_mapping/offline_mapping_noniid_global.py --seed=1248 --label_mapping=feature --log_dir=./logs/seed1248_noniid_gan_weight/GeFL_gan_pacfl_iid \
#     --mnist_split ./data/raw/splits/MNIST_C10_New0_alpha0p1_seed1248.json \
#     --emnist_split ./data/raw/splits/EMNIST_C10_New0_alpha0p1_seed1248.json \
#     --cifar10_split ./data/raw/splits/CIFAR10_C10_New0_alpha0p1_seed1248.json
    
# python label_mapping/offline_mapping_noniid_global.py --seed=15698 --label_mapping=feature --log_dir=./logs/seed15698_noniid_gan_weight/GeFL_gan_pacfl_iid \
#     --mnist_split ./data/raw/splits/MNIST_C10_New0_alpha0p1_seed15698.json \
#     --emnist_split ./data/raw/splits/EMNIST_C10_New0_alpha0p1_seed15698.json \
#     --cifar10_split ./data/raw/splits/CIFAR10_C10_New0_alpha0p1_seed15698.json

# python label_mapping/offline_mapping_noniid.py --seed=1 --label_mapping=temp2 --log_dir=./logs/seed1_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid.py --seed=42 --label_mapping=temp2 --log_dir=./logs/seed42_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid.py --seed=758 --label_mapping=temp2 --log_dir=./logs/seed758_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid.py --seed=1248 --label_mapping=temp2 --log_dir=./logs/seed1248_noniid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid.py --seed=15698 --label_mapping=temp2 --log_dir=./logs/seed15698_noniid_gan_weight/GeFL_gan_pacfl_iid


# ------------------------------------------------------------------------
# iid
# ------------------------------------------------------------------------

# 1) train + save checkpoints every 5 rounds -> logs/seed{S}_iid_gan_weight/GeFL_gan_pacfl_iid
#    (skipped only when the run finished: round-45 checkpoint exists)
for S in 1 42 758 1248 15698; do
    [ -f ./logs/seed${S}_iid_gan_weight/GeFL_gan_pacfl_iid/server_checkpoints_45.pth ] && continue
    python3 main.py --seed=$S --algorithm=GeFL_gan_pacfl_iid --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 \
        --device=$DEVICE --exp_conf=./configs/het-iid-exp.yaml --pacfl_cluster_alpha=10 --pacfl_basis_budget=20 \
        --exp_timestamp=seed${S}_iid_gan_weight
done

# 2) offline label mapping on the saved checkpoints
# python label_mapping/offline_mapping_noniid_global.py --seed=1 --label_mapping=improve_single --log_dir=./logs/seed1_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=42 --label_mapping=improve_single --log_dir=./logs/seed42_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=758 --label_mapping=improve_single --log_dir=./logs/seed758_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=1248 --label_mapping=improve_single --log_dir=./logs/seed1248_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=15698 --label_mapping=improve_single --log_dir=./logs/seed15698_iid_gan_weight/GeFL_gan_pacfl_iid


# python label_mapping/offline_mapping_noniid_global.py --seed=1 --label_mapping=improve_single_noniid --log_dir=./logs/seed1_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=42 --label_mapping=improve_single_noniid --log_dir=./logs/seed42_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=758 --label_mapping=improve_single_noniid --log_dir=./logs/seed758_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=1248 --label_mapping=improve_single_noniid --log_dir=./logs/seed1248_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=15698 --label_mapping=improve_single_noniid --log_dir=./logs/seed15698_iid_gan_weight/GeFL_gan_pacfl_iid


# python label_mapping/offline_mapping_noniid_global.py --seed=1 --label_mapping=missing_link --log_dir=./logs/seed1_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=42 --label_mapping=missing_link --log_dir=./logs/seed42_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=758 --label_mapping=missing_link --log_dir=./logs/seed758_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=1248 --label_mapping=missing_link --log_dir=./logs/seed1248_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=15698 --label_mapping=missing_link --log_dir=./logs/seed15698_iid_gan_weight/GeFL_gan_pacfl_iid

python3 label_mapping/offline_mapping_noniid_global.py --seed=1 --label_mapping=missing_link_single --log_dir=./logs/seed1_iid_gan_weight/GeFL_gan_pacfl_iid --device=$DEVICE
python3 label_mapping/offline_mapping_noniid_global.py --seed=42 --label_mapping=missing_link_single --log_dir=./logs/seed42_iid_gan_weight/GeFL_gan_pacfl_iid --device=$DEVICE
python3 label_mapping/offline_mapping_noniid_global.py --seed=758 --label_mapping=missing_link_single --log_dir=./logs/seed758_iid_gan_weight/GeFL_gan_pacfl_iid --device=$DEVICE
python3 label_mapping/offline_mapping_noniid_global.py --seed=1248 --label_mapping=missing_link_single --log_dir=./logs/seed1248_iid_gan_weight/GeFL_gan_pacfl_iid --device=$DEVICE
python3 label_mapping/offline_mapping_noniid_global.py --seed=15698 --label_mapping=missing_link_single --log_dir=./logs/seed15698_iid_gan_weight/GeFL_gan_pacfl_iid --device=$DEVICE

python3 label_mapping/offline_mapping_noniid_global.py --seed=1 --label_mapping=psi_trivial --log_dir=./logs/seed1_iid_gan_weight/GeFL_gan_pacfl_iid --device=$DEVICE
python3 label_mapping/offline_mapping_noniid_global.py --seed=42 --label_mapping=psi_trivial --log_dir=./logs/seed42_iid_gan_weight/GeFL_gan_pacfl_iid --device=$DEVICE
python3 label_mapping/offline_mapping_noniid_global.py --seed=758 --label_mapping=psi_trivial --log_dir=./logs/seed758_iid_gan_weight/GeFL_gan_pacfl_iid --device=$DEVICE
python3 label_mapping/offline_mapping_noniid_global.py --seed=1248 --label_mapping=psi_trivial --log_dir=./logs/seed1248_iid_gan_weight/GeFL_gan_pacfl_iid --device=$DEVICE
python3 label_mapping/offline_mapping_noniid_global.py --seed=15698 --label_mapping=psi_trivial --log_dir=./logs/seed15698_iid_gan_weight/GeFL_gan_pacfl_iid --device=$DEVICE


# python label_mapping/offline_mapping_noniid_global.py --seed=1 --label_mapping=image-cs --log_dir=./logs/seed1_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=42 --label_mapping=image-cs --log_dir=./logs/seed42_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=758 --label_mapping=image-cs --log_dir=./logs/seed758_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=1248 --label_mapping=image-cs --log_dir=./logs/seed1248_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid_global.py --seed=15698 --label_mapping=image-cs --log_dir=./logs/seed15698_iid_gan_weight/GeFL_gan_pacfl_iid


# python label_mapping/offline_mapping_noniid_global.py --seed=1 --label_mapping=feature --log_dir=./logs/seed1_iid_gan_weight/GeFL_gan_pacfl_iid \
#     --mnist_split ./data/raw/splits/MNIST_C10_New0_alpha100p0_seed1.json \
#     --emnist_split ./data/raw/splits/EMNIST_C10_New0_alpha100p0_seed1.json \
#     --cifar10_split ./data/raw/splits/CIFAR10_C10_New0_alpha100p0_seed1.json

# python label_mapping/offline_mapping_noniid_global.py --seed=42 --label_mapping=feature --log_dir=./logs/seed42_iid_gan_weight/GeFL_gan_pacfl_iid \
#     --mnist_split ./data/raw/splits/MNIST_C10_New0_alpha100p0_seed42.json \
#     --emnist_split ./data/raw/splits/EMNIST_C10_New0_alpha100p0_seed42.json \
#     --cifar10_split ./data/raw/splits/CIFAR10_C10_New0_alpha100p0_seed42.json

# python label_mapping/offline_mapping_noniid_global.py --seed=758 --label_mapping=feature --log_dir=./logs/seed758_iid_gan_weight/GeFL_gan_pacfl_iid \
#     --mnist_split ./data/raw/splits/MNIST_C10_New0_alpha100p0_seed758.json \
#     --emnist_split ./data/raw/splits/EMNIST_C10_New0_alpha100p0_seed758.json \
#     --cifar10_split ./data/raw/splits/CIFAR10_C10_New0_alpha100p0_seed758.json

# python label_mapping/offline_mapping_noniid_global.py --seed=1248 --label_mapping=feature --log_dir=./logs/seed1248_iid_gan_weight/GeFL_gan_pacfl_iid \
#     --mnist_split ./data/raw/splits/MNIST_C10_New0_alpha100p0_seed1248.json \
#     --emnist_split ./data/raw/splits/EMNIST_C10_New0_alpha100p0_seed1248.json \
#     --cifar10_split ./data/raw/splits/CIFAR10_C10_New0_alpha100p0_seed1248.json
    
# python label_mapping/offline_mapping_noniid_global.py --seed=15698 --label_mapping=feature --log_dir=./logs/seed15698_iid_gan_weight/GeFL_gan_pacfl_iid \
#     --mnist_split ./data/raw/splits/MNIST_C10_New0_alpha100p0_seed15698.json \
#     --emnist_split ./data/raw/splits/EMNIST_C10_New0_alpha100p0_seed15698.json \
#     --cifar10_split ./data/raw/splits/CIFAR10_C10_New0_alpha100p0_seed15698.json



# python label_mapping/offline_mapping_noniid.py --seed=1 --label_mapping=temp2 --log_dir=./logs/seed1_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid.py --seed=42 --label_mapping=temp2 --log_dir=./logs/seed42_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid.py --seed=758 --label_mapping=temp2 --log_dir=./logs/seed758_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid.py --seed=1248 --label_mapping=temp2 --log_dir=./logs/seed1248_iid_gan_weight/GeFL_gan_pacfl_iid
# python label_mapping/offline_mapping_noniid.py --seed=15698 --label_mapping=temp2 --log_dir=./logs/seed15698_iid_gan_weight/GeFL_gan_pacfl_iid


TOTAL_TIME=$((SECONDS - TOTAL_START))
echo "Total execution time: $((TOTAL_TIME / 3600)) hours, $((TOTAL_TIME / 60)) minutes and $((TOTAL_TIME % 60)) seconds."