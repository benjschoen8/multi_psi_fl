"""
New-client accuracy comparison, PSI included (utils/plot_comparison.py untouched).

Same CLI as utils/plot_comparison.py, but also reads the CURRENT new-client output
(<Dataset>_newclient_history.csv, columns Epoch,Accuracy) and whole folders:

    python utils/plot_comparison_new.py \
        --csvs "PSI (filter):logs/psi_filter/GeFL_gan_pacfl_iid/new_client_nc/our_finetune_global_single_dataset" \
               "Missing Link:logs/missing_link/GeFL_gan_pacfl_iid/new_client_nc/our_finetune_global_single_dataset" \
               "baseline:logs/psi_filter/GeFL_gan_pacfl_iid/new_client_nc/baseline_single_dataset" \
        --output_dir plot/new_client_psi

  label:path   path = a folder (all *_newclient_history.csv inside), one *_newclient_history.csv,
               or an old-style accuracy_log.csv (dataset, model, epoch, combined_acc).
Labels containing "PSI" get color slot 1 and a dashed line. One figure per dataset + summary.txt.
"""
import argparse
import glob
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plot"))
from plot_rt_common_new import TEXT2, colors, plt  # noqa: E402


def load(label, path):
    files = sorted(glob.glob(os.path.join(path, "*_newclient_history.csv"))) if os.path.isdir(path) else [path]
    out = []
    for f in files:
        if not os.path.exists(f):
            print(f"[Warning] not found: {f}")
            continue
        df = pd.read_csv(f)
        if "combined_acc" in df.columns:                           # old accuracy_log.csv
            df = df.rename(columns={"epoch": "Epoch", "combined_acc": "Accuracy"})
            df["dataset"] = df["dataset"] + (" - " + df["model"] if "model" in df else "")
        else:
            df["dataset"] = os.path.basename(f).replace("_newclient_history.csv", "")
        df["label"] = label
        out.append(df[["dataset", "Epoch", "Accuracy", "label"]])
        print(f"  {label:<20} <- {f} ({len(df)} rows)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True, help='"label:path" (path = folder or csv)')
    ap.add_argument("--output_dir", default="plot/new_client_psi")
    a = ap.parse_args()
    out = os.path.join(a.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "csv_sources.txt"), "w") as f:
        f.write("\n".join(a.csvs) + "\n")

    parts = []
    for spec in a.csvs:
        label, path = spec.split(":", 1)
        parts += load(label, path)
    if not parts:
        sys.exit("[Error] no data loaded")
    df = pd.concat(parts, ignore_index=True)
    df["Accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")
    labels = list(dict.fromkeys(df["label"]))
    psi = {l for l in labels if "psi" in l.lower()}
    cmap = colors(labels, psi)

    lines = []
    for ds in sorted(df["dataset"].unique()):
        fig, ax = plt.subplots(figsize=(8, 5))
        lines += [f"\n== {ds} ==", f"  {'method':<24}{'final':>9}{'best':>9}"]
        for l in labels:
            d = df[(df["dataset"] == ds) & (df["label"] == l)].sort_values("Epoch")
            if d.empty:
                continue
            best, final = d["Accuracy"].max(), d["Accuracy"].iloc[-1]
            ax.plot(d["Epoch"], d["Accuracy"], color=cmap[l], linestyle="--" if l in psi else "-",
                    marker="o", markevery=max(1, len(d) // 15), label=f"{l}  (final {final:.2f}%, best {best:.2f}%)")
            lines.append(f"  {l:<24}{final:>8.2f}%{best:>8.2f}%")
        ax.set_xlabel("New-client epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(left=0)
        ax.set_title(f"New client  |  {ds}", color=TEXT2, fontsize=11, loc="left")
        ax.legend(fontsize=9, loc="lower right")
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"{ds}.{ext}"), bbox_inches="tight", dpi=150)
        plt.close(fig)
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
