"""
Label-mapping metrics vs threshold, existing methods + PSI (RT) as a horizontal reference line.
(plot_label_mapping_method_acc.py untouched.)

PSI has no entropy / threshold knob -- it is computed once before training -- so it is drawn as a
flat line across every panel.

  python plot/plot_label_mapping_method_acc_new.py \
      --method "Improve=label_mapping/pacfl_3cluster/noniid/improve_single_seed15698/label_mapping/offline_improve_single_noniid_mapping_acc.csv" \
      --method "Missing Link=label_mapping/.../offline_missing_link_noniid_mapping_acc.csv::missing_threshold" \
      --psi    "PSI (filter)=logs/rt_filter/GeFL_gan_pacfl_iid/rt_mapping_acc.csv" \
      --round 25 --out plot/label_mapping_psi

  --method NAME=CSV[::xcol]  xcol defaults to entropy_ratio
  --psi    NAME=rt_mapping_acc.csv   (repeatable, e.g. filter vs affscan vs attn; group-level row used)
"""
import argparse
import os

import pandas as pd

from plot_rt_common_new import TEXT2, colors, parse_named, plt

METRICS = {"Recall": "recall", "Specificity": "specificity", "Precision": "precision",
           "Average_Accuracy": "average_accuracy", "F1_Score": "f1_score", "MCC": "mcc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", default=[])
    ap.add_argument("--psi", action="append", default=[])
    ap.add_argument("--round", type=int, default=None, help="global_round to plot; default: every round found")
    ap.add_argument("--level", default="group", choices=["group", "client"])
    ap.add_argument("--out", default="plot/label_mapping_psi")
    a = ap.parse_args()

    methods = [(n, pd.read_csv(p), x or "entropy_ratio") for n, p, x in parse_named(a.method)]
    psis = []
    for n, p, _ in parse_named(a.psi):
        df = pd.read_csv(p)
        psis.append((n, df[df["level"] == a.level].iloc[0]))
    cmap = colors([n for n, *_ in methods] + [n for n, _ in psis], {n for n, _ in psis})

    rounds = [a.round] if a.round is not None else sorted(
        set.union(set(), *[set(df["global_round"].unique()) for _, df, _ in methods]) or {0})

    for mname, col in METRICS.items():
        os.makedirs(os.path.join(a.out, mname), exist_ok=True)
        for rnd in rounds:
            fig, ax = plt.subplots(figsize=(8, 5))
            for n, df, x in methods:
                d = df[df["global_round"] == rnd].sort_values(x) if "global_round" in df else df.sort_values(x)
                if d.empty or x not in d:
                    print(f"skip {n}: no round {rnd} / column {x}")
                    continue
                ax.plot(d[x], d[col], marker="o", color=cmap[n], label=n)
            for n, row in psis:
                ax.axhline(row[col], color=cmap[n], linestyle="--", label=f"{n} (before training)")
            ax.set_xlabel("Entropy ratio / threshold (baselines only)")
            ax.set_ylabel(mname)
            ax.set_ylim(-1.05, 1.05) if mname == "MCC" else ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{mname}  |  baselines at global round {rnd}", color=TEXT2, fontsize=11, loc="left")
            ax.legend(loc="best", fontsize=9)
            path = os.path.join(a.out, mname, f"{mname}_Round_{rnd}.pdf")
            fig.savefig(path, bbox_inches="tight")
            fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
            plt.close(fig)
            print("saved", path)


if __name__ == "__main__":
    main()
