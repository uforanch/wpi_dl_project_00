"""Label-efficiency curve from probe.csv files in several folders.

    python plot_curve.py dsert_stability/crossmae/s5_curve_* --out curve_crossmae.png

Plots mean ± std over all runs (seeds) of val mAP50 and Vehicle AP against the
fraction of labelled training frames, one line per experiment. Collapsed runs
(train mAP50 below 0.15 at 100 % labels, below 0.45 otherwise) are left out and
counted in the legend. Also prints the numbers as a table.
"""
import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {"a_random_init": "random init", "a_imagenet": "ImageNet (no pretraining)",
         "b_plain": "pretrained, plain loss", "c_ego": "pretrained, ego-weighted",
         "e2d_weighted": "pretrained e2d, weighted"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+")
    p.add_argument("--out", default="label_efficiency.png")
    p.add_argument("--keep", nargs="*", default=["a_random_init"], metavar="EXP",
                   help="experiments whose low-train-mAP runs are kept, not dropped as collapsed "
                        "(default: a_random_init, whose features are simply weak)")
    a = p.parse_args()
    files = []
    for f in a.folders:
        files += glob.glob(os.path.join(f, "probe.csv"))
    if not files:
        raise SystemExit("no probe.csv in " + " ".join(a.folders))
    data = defaultdict(lambda: defaultdict(list))
    dropped = defaultdict(int)
    for fn in files:
        with open(fn, newline="") as fh:
            for r in csv.DictReader(fh):
                fr = float(r["fraction"])
                thr = 0.15 if fr >= 1.0 else 0.45
                if r.get("diverged_step") or (float(r["train_mAP50"]) < thr
                                              and r["experiment"] not in a.keep):
                    dropped[r["experiment"]] += 1
                    continue
                data[r["experiment"]][fr].append((float(r["mAP50"]),
                                                  float(r.get("AP50_Vehicle") or "nan")))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    print("| experiment | labels | runs | val mAP50 | Vehicle AP |\n|---|---|---|---|---|")
    for exp in sorted(data):
        fr = sorted(data[exp])
        for k, ax in enumerate(axs):
            m = [statistics.mean(v[k] for v in data[exp][f]) for f in fr]
            s = [statistics.stdev(v[k] for v in data[exp][f]) if len(data[exp][f]) > 1 else 0
                 for f in fr]
            lab = NAMES.get(exp, exp) + (f" ({dropped[exp]} collapsed)" if dropped[exp] else "")
            ax.errorbar([100 * f for f in fr], m, yerr=s, marker="o", capsize=3, label=lab)
        for f in fr:
            v = data[exp][f]
            ms = lambda i: (f"{statistics.mean(x[i] for x in v):.3f}" +
                            (f" ± {statistics.stdev(x[i] for x in v):.3f}" if len(v) > 1 else ""))
            print(f"| {exp} | {f:.0%} | {len(v)} | {ms(0)} | {ms(1)} |")
    for ax, t in zip(axs, ["val mAP50", "Vehicle AP50"]):
        ax.set_xscale("log")
        ax.set_xticks([10, 25, 50, 100])
        ax.set_xticklabels(["10%", "25%", "50%", "100%"])
        ax.set_xlabel("labelled training frames")
        ax.set_ylabel(t)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.suptitle("Label efficiency (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
