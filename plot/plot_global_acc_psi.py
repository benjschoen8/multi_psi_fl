"""Global inference model accuracy per round, one seed: class_name (oracle), image-bi, feature-bi, image-cs, psi_trivial.
"mix" = plain mean of the MNIST / EMNIST / CIFAR10 accuracies (same as plot/write_mix_acc.py -> mix_new.csv).
Reads the global_model_acc_*.csv files written by trainer/GeFL_gan_pacfl_iid/server.py (see run_psi_compare.sh).
usage: python3 plot/plot_global_acc_psi.py --seed 15698
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=15698)
p.add_argument("--out", default="./plot/global_accuracy_plots/psi_compare")
a = p.parse_args()

runs = {
    "Oracle (class names)": f"logs/seed{a.seed}_iid_class_name/GeFL_gan_pacfl_iid",
    "Image-bi": f"logs/seed{a.seed}_iid_gan_weight/GeFL_gan_pacfl_iid",
    "Feature-bi": f"logs/seed{a.seed}_iid_feature_bi/GeFL_gan_pacfl_iid",
    "Image-cs": f"logs/seed{a.seed}_iid_image_cs/GeFL_gan_pacfl_iid",
    "PSI-Trivial": f"logs/seed{a.seed}_iid_psi_trivial/GeFL_gan_pacfl_iid",
}
DATASETS = ["MNIST", "EMNIST", "CIFAR10"]


def load(d, ds):
    """Round -> accuracy for one run; ds == "mix" averages the three datasets per round."""
    if ds == "mix":
        parts = [load(d, x) for x in DATASETS]
        if any(p is None for p in parts):
            return None
        return pd.concat(parts, axis=1, join="inner").mean(axis=1)
    f = os.path.join(d, f"global_model_acc_{ds}.csv")
    if not os.path.exists(f):
        print("missing", f)
        return None
    return pd.read_csv(f).groupby("Round")["Accuracy"].mean()

os.makedirs(a.out, exist_ok=True)

for ds in ["mix"] + DATASETS:
    plt.figure(figsize=(10, 6))
    for name, d in runs.items():
        acc = load(d, ds)
        if acc is not None:
            plt.plot(acc.index, acc.values, marker="o", ms=4, linewidth=2, label=name)
    plt.title(f"Global model accuracy ({ds}), seed {a.seed}")
    plt.xlabel("Global round")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    path = os.path.join(a.out, f"global_acc_{ds}_seed{a.seed}.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("saved", path)
