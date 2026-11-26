"""Summarise probe-stability runs for the report.

    python probe_report.py dsert_stability/videomae --logs slurm-*.out
    python probe_report.py dsert_stability/crossmae/final_5gb dsert_stability/crossmae/final_20gb \
        --logs slurm-*.out --paired a_imagenet --md final_report.md

Each folder is one setting (it holds a probe.csv); a folder without probe.csv
is searched one level down, so a whole sweep can be given at once.

Tables:
  1. per setting / experiment / fraction: runs, collapsed, diverged, mean +- std
     of val mAP50, Vehicle AP and train mAP50 (all runs, and healthy runs only)
  2. (with --logs) how each run's loss behaved: diverged / never started /
     spiked / learned -- the Step 2 classification
  3. (with --paired REF) difference to REF on the same seed and fraction

Collapsed = diverged (NaN loss), or train mAP50 below --collapse-full at 100%
labels / below --collapse-sub at smaller fractions (defaults 0.15 / 0.45, the
thresholds used in the earlier notes).
Only the Python standard library is needed.
"""
import argparse
import csv
import glob
import math
import os
import re
import statistics
from collections import defaultdict


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def mean_std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}"


def find_folders(paths):
    out = []
    for p in paths:
        if os.path.isfile(os.path.join(p, "probe.csv")):
            out.append(p)
        else:
            out += sorted(os.path.dirname(f) for f in glob.glob(os.path.join(p, "*", "probe.csv")))
    return [os.path.abspath(f) for f in out]


def load_rows(folders):
    rows = []
    for f in folders:
        with open(os.path.join(f, "probe.csv"), newline="") as fh:
            for r in csv.DictReader(fh):
                r["_folder"] = f
                rows.append(r)
    return rows


def is_collapsed(r, a):
    if r.get("diverged_step"):
        return True
    thr = a.collapse_full if fnum(r["fraction"]) >= 1.0 else a.collapse_sub
    return fnum(r["train_mAP50"]) < thr


# ---------------------------------------------------------------- log parsing
MARK = re.compile(r"^### SETTING (\S+) SEED (\d+) FLAGS")
LOSS = re.compile(r"^(\S+) (\d+)% step (\d+)/(\d+) loss (\S+) \[(.*?)\](.*)$")


def parse_logs(files):
    """-> {(folder, seed, experiment, fraction): dict(steps, losses, cls, diverged)}"""
    runs = {}
    for path in files:
        cur = None
        with open(path, errors="replace") as fh:
            for line in fh:
                m = MARK.match(line)
                if m:
                    cur = (os.path.abspath(m.group(1)), int(m.group(2)))
                    continue
                m = LOSS.match(line)
                if not m or cur is None:
                    continue
                exp, pct, step, _, loss, detail, tail = m.groups()
                key = cur + (exp, round(int(pct) / 100, 2))
                if int(step) == 1:                       # a (re)started run
                    runs[key] = dict(steps=[], losses=[], cls=[], diverged=False)
                rec = runs.setdefault(key, dict(steps=[], losses=[], cls=[], diverged=False))
                parts = detail.split()
                comp = {parts[i]: fnum(parts[i + 1]) for i in range(0, len(parts) - 1, 2)}
                rec["steps"].append(int(step))
                rec["losses"].append(fnum(loss))
                rec["cls"].append(comp.get("classification", float("nan")))
                if "DIVERGED" in tail or not math.isfinite(fnum(loss)):
                    rec["diverged"] = True
    return runs


def classify(rec):
    """Heuristic Step-2 label. Check the numbers by eye for the report."""
    if rec["diverged"]:
        return "diverged"
    L = [x for x in rec["losses"] if math.isfinite(x)]
    if len(L) < 4:
        return "too few log points"
    first, best = L[0], min(L)
    end = statistics.mean(L[-3:])
    if best >= 0.85 * first:                     # never got clearly below the start
        return "never started"
    run_min, spiked = L[0], False
    for x in L[1:]:
        if x > 2 * run_min and run_min < 0.85 * first:
            spiked = True
        run_min = min(run_min, x)
    if spiked and end > 1.3 * best:
        return "spiked"
    return "learned"


# -------------------------------------------------------------------- output
def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def short(folder):
    parts = folder.split(os.sep)
    return "/".join(parts[-2:])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+")
    p.add_argument("--logs", nargs="*", default=[], help="slurm-*.out files of the runs")
    p.add_argument("--paired", default=None, help="reference experiment, e.g. a_imagenet")
    p.add_argument("--collapse-full", type=float, default=0.15)
    p.add_argument("--collapse-sub", type=float, default=0.45)
    p.add_argument("--md", default=None, help="also write the tables to this markdown file")
    a = p.parse_args()

    folders = find_folders(a.folders)
    if not folders:
        raise SystemExit("no probe.csv found in " + " ".join(a.folders))
    rows = load_rows(folders)
    logs = parse_logs([f for f in a.logs if os.path.isfile(f)])
    sections = []

    # 1. per setting summary
    groups = defaultdict(list)
    for r in rows:
        groups[(r["_folder"], r["experiment"], fnum(r["fraction"]))].append(r)
    t1 = []
    for (f, exp, frac), rs in sorted(groups.items()):
        ok = [r for r in rs if not is_collapsed(r, a)]
        t1.append([short(f), exp, f"{frac:.0%}", len(rs),
                   sum(is_collapsed(r, a) for r in rs),
                   sum(bool(r.get("diverged_step")) for r in rs),
                   mean_std([fnum(r["mAP50"]) for r in rs]),
                   mean_std([fnum(r["mAP50"]) for r in ok]),
                   mean_std([fnum(r.get("AP50_Vehicle")) for r in ok]),
                   mean_std([fnum(r["train_mAP50"]) for r in rs])])
    tot = len(rows)
    col = sum(is_collapsed(r, a) for r in rows)
    sections.append("## Collapse rate and scores per setting\n\n"
                    f"{col} of {tot} runs collapsed overall.\n\n" +
                    table(["setting", "experiment", "labels", "runs", "collapsed", "diverged",
                           "val mAP50 (all)", "val mAP50 (healthy)", "Vehicle AP (healthy)",
                           "train mAP50 (all)"], t1))

    # 2. loss behaviour per run (Step 2)
    if logs:
        by_row = {(r["_folder"], int(fnum(r["seed"])), r["experiment"], round(fnum(r["fraction"]), 2)): r
                  for r in rows}
        t2, counts = [], defaultdict(lambda: defaultdict(int))
        for key, rec in sorted(logs.items()):
            if key[0] not in folders:
                continue
            lab = classify(rec)
            r = by_row.get(key)
            counts[(short(key[0]), key[2], key[3])][lab] += 1
            L = [x for x in rec["losses"] if math.isfinite(x)]
            C = [x for x in rec["cls"] if math.isfinite(x)]
            t2.append([short(key[0]), key[2], f"{key[3]:.0%}", key[1], lab,
                       f"{L[0]:.3f}" if L else "-", f"{min(L):.3f}" if L else "-",
                       f"{L[-1]:.3f}" if L else "-",
                       f"{C[0]:.3f} → {C[-1]:.3f}" if C else "-",
                       ("COLLAPSED" if is_collapsed(r, a) else "ok") if r else "not in csv"])
        sections.append("## Loss behaviour per run (from the logs)\n\n"
                        "Labels are a heuristic: *diverged* = NaN/inf loss; *never started* = "
                        "loss never fell below 85% of its first value; *spiked* = loss jumped to over 2× its "
                        "lowest value and ended >30% above it; otherwise *learned*. "
                        "Check the numbers before quoting a label.\n\n" +
                        table(["setting", "experiment", "labels", "seed", "behaviour",
                               "first loss", "min loss", "last loss", "classification loss",
                               "csv"], t2))
        t2b = [[s, e, f"{fr:.0%}"] + [c.get(k, 0) for k in
               ("diverged", "never started", "spiked", "learned")]
               for (s, e, fr), c in sorted(counts.items())]
        sections.append("### Behaviour counts\n\n" + table(
            ["setting", "experiment", "labels", "diverged", "never started", "spiked", "learned"], t2b))

    # 3. paired comparison with a reference experiment
    if a.paired:
        # reference from the SAME folder (same settings); a folder without any
        # reference runs (e.g. final_20gb) borrows them from the other folders
        ref, ref_any, has_ref = defaultdict(list), defaultdict(list), set()
        for r in rows:
            if r["experiment"] == a.paired:
                has_ref.add(r["_folder"])
                if not is_collapsed(r, a):
                    k = (round(fnum(r["fraction"]), 2), int(fnum(r["seed"])))
                    ref[(r["_folder"],) + k].append(r)
                    ref_any[k].append(r)
        t3 = []
        for (f, exp, frac), rs in sorted(groups.items()):
            if exp == a.paired:
                continue
            d_map, d_veh, wins, n = [], [], 0, 0
            for r in rs:
                k = (round(frac, 2), int(fnum(r["seed"])))
                rr = ref.get((f,) + k) if f in has_ref else ref_any.get(k)
                if not rr or is_collapsed(r, a):
                    continue
                n += 1
                dm = fnum(r["mAP50"]) - statistics.mean(fnum(x["mAP50"]) for x in rr)
                d_map.append(dm)
                d_veh.append(fnum(r.get("AP50_Vehicle")) -
                             statistics.mean(fnum(x.get("AP50_Vehicle")) for x in rr))
                wins += dm > 0
            t3.append([short(f), exp, f"{frac:.0%}", n, mean_std(d_map), mean_std(d_veh),
                       f"{wins}/{n}"])
        sections.append(f"## Difference to {a.paired} (same seed and labels, healthy runs only)\n\n"
                        "Positive = better than the reference.\n\n" + table(
                            ["setting", "experiment", "labels", "pairs", "Δ val mAP50",
                             "Δ Vehicle AP", "beats ref"], t3))

    text = "\n\n".join(sections) + "\n"
    print(text)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write("# Probe stability report\n\n" + text)
        print(f"written to {a.md}")


if __name__ == "__main__":
    main()
