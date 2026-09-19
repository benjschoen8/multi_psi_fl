"""
RT diagnostics from one run: python plot/plot_rt_diagnostics_new.py logs/<ts>/GeFL_gan_pacfl_iid

Figures (PDF + PNG) in <log_dir>/rt_diag_plots/:
  cos_hist_<signal>.pdf     same-class vs diff-class cosine; ladder rungs drawn -> choose the floor
  class_counts.pdf          local samples per (client, class); rt_min_samples marked
  he_mismatch.pdf           CKKS vs plaintext bit disagreement as a function of |v - delta| (D15)
  v_est_err.pdf             what receiver j can infer about v from masked values (D14)
  summary.txt               ties, candidates, metrics, suggested floor
"""
import json
import os
import sys

import numpy as np

from plot_rt_common_new import SERIES, TEXT2, plt


def save(fig, d, name):
    fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(d, name + ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("saved", os.path.join(d, name + ".pdf"))


def main(log_dir):
    Z = np.load(os.path.join(log_dir, "rt_diagnostics.npz"))
    J = json.load(open(os.path.join(log_dir, "rt_diagnostics.json")))
    out = os.path.join(log_dir, "rt_diag_plots")
    os.makedirs(out, exist_ok=True)
    ladders = J["ladder"] if isinstance(J["ladder"], dict) else None
    ladder_of = lambda sig: np.array(ladders.get(sig, ladders.get("default", [])) if ladders else J["ladder"])
    lines = [f"method={J['method']} psi={J['psi']} anchors={J['n_anchors']}",
             f"classes below rt_min_samples: {J['classes_below_min']}",
             f"candidates (Alg.2): {J.get('candidates', '-')}   filtered out by image affinity (Alg.5): {J.get('filtered_out', '-')}"]
    lines += [f"{k}: {v}" for k, v in J.items() if k.endswith("ties")]

    # ---- cosine histograms per signal
    for sig in sorted({k.rsplit("_cos_", 1)[0] for k in Z.files if "_cos_" in k}):
        same, diff = Z.get(f"{sig}_cos_same"), Z.get(f"{sig}_cos_diff")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bins = np.linspace(-1, 1, 81)
        if diff is not None and len(diff):
            ax.hist(diff, bins=bins, density=True, color=SERIES[1], alpha=0.55, label=f"different class (n={len(diff)})")
        if same is not None and len(same):
            ax.hist(same, bins=bins, density=True, color=SERIES[0], alpha=0.55, label=f"same class (n={len(same)})")
        for d in ladder_of(sig):
            ax.axvline(d, color=TEXT2, linewidth=0.6, linestyle=":")
        if same is not None and diff is not None and len(same) and len(diff):
            grid = np.linspace(-1, 1, 401)
            err = [(diff >= t).mean() + (same < t).mean() for t in grid]
            t = grid[int(np.argmin(err))]
            ax.axvline(t, color=SERIES[2], linewidth=2, label=f"min-error floor ~ {t:.2f}")
            lines.append(f"[{sig}] suggested ladder floor ~ {t:.2f}  "
                         f"(same median {np.median(same):.2f}, diff 99th pct {np.percentile(diff, 99):.2f})")
        ax.set_xlabel("cosine  <s_i[a], s_j[b]>   (dotted = ladder rungs)")
        ax.set_ylabel("density")
        ax.set_title(f"signal: {sig}", color=TEXT2, fontsize=11, loc="left")
        ax.legend(fontsize=9)
        save(fig, out, f"cos_hist_{sig}")

    # ---- per-class sample counts
    c = Z["class_counts"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.log10(np.maximum(c, 1)), bins=40, color=SERIES[0])
    ax.axvline(np.log10(10), color=SERIES[1], label="default rt_min_samples = 10")
    ax.set_xlabel("log10(local samples per client-class)")
    ax.set_ylabel("count")
    ax.legend(fontsize=9)
    save(fig, out, "class_counts")

    # ---- CKKS vs plaintext bit mismatch
    he = [k[:-len("he_dist")] for k in Z.files if k.endswith("he_dist")]
    if he:
        fig, ax = plt.subplots(figsize=(8, 4))
        for k, sig in enumerate(he):
            dist, mis = np.maximum(Z[sig + "he_dist"], 1e-9), Z[sig + "he_mismatch"]   # exact ties -> first bin
            edges = np.logspace(-9.01, 0, 37)
            idx = np.digitize(dist, edges)
            rate = [mis[idx == b].mean() if (idx == b).any() else np.nan for b in range(1, len(edges))]
            ax.semilogx(edges[1:], rate, marker="o", color=SERIES[k], label=f"{sig.rstrip('_')} (overall {mis.mean():.2e})")
            lines.append(f"[{sig.rstrip('_')}] CKKS/plain bit mismatch rate {mis.mean():.3e} over {len(mis)} slots")
        ax.set_xlabel("|v - nearest rung delta|")
        ax.set_ylabel("rank mismatch rate")
        ax.legend(fontsize=9)
        save(fig, out, "he_mismatch")

    # ---- v estimation error (leakage to receiver)
    ve = [k for k in Z.files if k.endswith("v_est_err")]
    if ve:
        fig, ax = plt.subplots(figsize=(8, 4))
        for k, key in enumerate(ve):
            e = Z[key]
            ax.hist(e, bins=np.linspace(-0.5, 0.5, 101), color=SERIES[k], alpha=0.55,
                    label=f"{key[:-len('_v_est_err')]}  |err| median {np.median(np.abs(e)):.3f}")
            lines.append(f"[{key}] receiver estimate of v: |err| median {np.median(np.abs(e)):.4f}, "
                         f"90th pct {np.percentile(np.abs(e), 90):.4f}, within +-0.02: {(np.abs(e) < 0.02).mean():.1%}")
        ax.set_xlabel("v_hat - v   (receiver's estimate from masked values)")
        ax.set_ylabel("count")
        ax.legend(fontsize=9)
        save(fig, out, "v_est_err")

    for lvl in ("client", "group"):
        m = J.get(f"metrics_{lvl}")
        if m:
            lines.append(f"[{lvl}] F1={m['f1_score']:.4f} MCC={m['mcc']:.4f} P={m['precision']:.4f} "
                         f"R={m['recall']:.4f} TP={m['TP']} FP={m['FP']} FN={m['FN']}")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
