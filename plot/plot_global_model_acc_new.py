"""
Global inference model accuracy per round, existing runs + PSI (RT) runs. (plot_global_model_acc.py untouched.)

  python plot/plot_global_model_acc_new.py \
      --run "Ours=logs/start25_noniid_gan_ours/GeFL_gan_pacfl_iid" \
      --run "Missing Link=logs/start25_noniid_gan_missinglink/GeFL_gan_pacfl_iid" \
      --psi "PSI (filter)=logs/rt_filter/GeFL_gan_pacfl_iid" \
      --datasets MNIST EMNIST CIFAR10 --out plot/global_accuracy_psi

Reads <run>/global_model_acc_<ds>.csv. "mix" = mean over --datasets per round (same as write_mix_acc.py).
PSI runs count unmapped test samples as WRONG (baselines skipped them); their coverage is plotted
in a separate figure, never on a second y-axis.
"""
import argparse
import os

import pandas as pd

from plot_rt_common_new import TEXT2, colors, parse_named, plt


def load(run, ds):
    p = os.path.join(run, f"global_model_acc_{ds}.csv")
    if not os.path.exists(p):
        print("missing", p)
        return None
    df = pd.read_csv(p)
    df = df.rename(columns={"global_round": "Round", "accuracy": "Accuracy"})
    return df.groupby("Round", as_index=False).last()      # last epoch row per round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--psi", action="append", default=[])
    ap.add_argument("--datasets", nargs="+", default=["MNIST", "EMNIST", "CIFAR10"])
    ap.add_argument("--out", default="plot/global_accuracy_psi")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = [(n, p) for n, p, _ in parse_named(a.run)] + [(n, p) for n, p, _ in parse_named(a.psi)]
    psi_names = {n for n, _, _ in parse_named(a.psi)}
    cmap = colors([n for n, _ in runs], psi_names)
    data = {n: {ds: load(p, ds) for ds in a.datasets} for n, p in runs}

    for ds in a.datasets + ["mix"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for n, _ in runs:
            if ds == "mix":
                parts = [d.set_index("Round")["Accuracy"] for d in data[n].values() if d is not None]
                if len(parts) != len(a.datasets):
                    continue
                s = pd.concat(parts, axis=1).dropna().mean(axis=1)
                x, y = s.index, s.values
            else:
                d = data[n][ds]
                if d is None:
                    continue
                x, y = d["Round"], d["Accuracy"]
            ax.plot(x, y, marker="o", color=cmap[n], linestyle="--" if n in psi_names else "-", label=n)
        ax.set_xlabel("Global round")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"Global model accuracy  |  {ds}", color=TEXT2, fontsize=11, loc="left")
        ax.legend(loc="best", fontsize=9)
        path = os.path.join(a.out, f"Global_Acc_{ds}.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print("saved", path)

    # coverage of PSI runs (fraction of test samples whose label has a global id)
    cov = [(n, ds, data[n][ds]) for n in psi_names for ds in a.datasets
           if data[n][ds] is not None and "Coverage" in data[n][ds]]
    if cov:
        fig, ax = plt.subplots(figsize=(8, 4))
        for k, (n, ds, d) in enumerate(cov):
            ax.plot(d["Round"], d["Coverage"], color=cmap[n], alpha=1 - 0.25 * (k % 3),
                    label=f"{n} | {ds}")
        ax.set_xlabel("Global round")
        ax.set_ylabel("Coverage (%)")
        ax.set_ylim(0, 105)
        ax.set_title("PSI runs: test samples with a mapped label", color=TEXT2, fontsize=11, loc="left")
        ax.legend(loc="best", fontsize=9)
        path = os.path.join(a.out, "Coverage_psi.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print("saved", path)


if __name__ == "__main__":
    main()
