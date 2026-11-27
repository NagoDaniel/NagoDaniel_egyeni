"""Population diversity of the GA under each mutation profile. Run from the project root:

  .venv/bin/python src/hp_diversity.py run --seeds 10 --out results/diversity.csv --workers 9
  .venv/bin/python src/hp_diversity.py stats results/diversity.csv
  .venv/bin/python src/hp_plots.py diversity results/diversity.csv      (plots, see hp_plots.py)

The GA runs for its whole generation budget (no early stop at the target), so every run of a
cell has the same length and the curves are comparable. Seeds and settings are those of the
final experiment, so a run here is the same run as in results/final.csv until it hits the target.

CSV: one row per run.
  hamming     mean pairwise Hamming distance / string length of the population, every STEP generations
  unique      share of distinct move strings in the population, same generations
  positions   final per-position distance (one value per move)
  energy      final energy of the best fold

stats writes next to the CSV:
  <prefix>_stats.csv   median over runs per (dim, length, profile): diversity averaged over the run,
                       diversity at the last generation, unique share at the end, and the final
                       diversity of the first and last third of the move positions
  <prefix>_tests.csv   every profile against uniform (Mann-Whitney, Holm over all tests), on the
                       run-average and on the final diversity
"""
import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from hp_ga import run_ga, PROFILES
from hp_stats import mwu, holm, write_csv, r2
from run_experiment import SEQS, TARGETS, ALPHABETS, GENERATIONS

STEP = 5   # keep every 5th generation (all budgets are multiples of 5)
FIELDS = ["profile", "dim", "seq", "seed", "energy", "hamming", "unique", "positions"]


def one_run(job):
    profile, dim, seq, seed, generations = job
    r = run_ga(seq, ALPHABETS[dim], generations=generations, mutation_profile=profile, seed=seed,
               track_diversity=True)
    kept = r["diversity"][::STEP]
    return {"profile": profile, "dim": dim, "seq": seq, "seed": seed, "energy": r["energy"],
            "hamming": ";".join(f"{h:.4f}" for h, _ in kept),
            "unique": ";".join(f"{u:.4f}" for _, u in kept),
            "positions": ";".join(f"{x:.4f}" for x in r["positions"])}


def run(args):
    seqs = [s for s in SEQS if not args.lengths or len(s) in args.lengths]
    jobs = [(p, d, s, seed, GENERATIONS[len(s)]) for d in args.dims for s in seqs for p in PROFILES
            for seed in range(args.seed_start, args.seed_start + args.seeds)]
    start = time.time()
    with open(args.out, "w", newline="") as f, Pool(args.workers) as pool:
        out = csv.DictWriter(f, FIELDS)
        out.writeheader()
        for k, row in enumerate(pool.imap(one_run, jobs, chunksize=1), 1):
            out.writerow(row)
            if k % 20 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} runs, {time.time() - start:.0f}s", flush=True)


# ---------- reading and statistics ----------
def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({"profile": r["profile"], "dim": int(r["dim"]), "seq": r["seq"], "n": len(r["seq"]),
                     "seed": int(r["seed"]), "energy": int(r["energy"]) if r["energy"] else None,
                     "hamming": [float(x) for x in r["hamming"].split(";")],
                     "unique": [float(x) for x in r["unique"].split(";")],
                     "positions": [float(x) for x in r["positions"].split(";")]})
    return rows


def metrics(r):
    """Numbers of one run: diversity averaged over the run, at the end, unique share at the end,
    final diversity of the first third and of the last third of the move positions."""
    pos, third = r["positions"], max(1, len(r["positions"]) // 3)
    return {"mean": float(np.mean(r["hamming"])), "final": r["hamming"][-1], "unique": r["unique"][-1],
            "head": float(np.mean(pos[:third])), "tail": float(np.mean(pos[-third:]))}


def cells(rows):
    return sorted({(r["dim"], r["n"]) for r in rows})


def profiles_in(rows):
    return [p for p in PROFILES if any(r["profile"] == p for r in rows)]


def stats(path):
    rows, stem = load(path), Path(path).with_suffix("")
    for r in rows:
        r["m"] = metrics(r)
    table, tests = [], []
    for dim, n in cells(rows):
        sel = {p: [r["m"] for r in rows if (r["dim"], r["n"], r["profile"]) == (dim, n, p)]
               for p in profiles_in(rows)}
        for p, ms in sel.items():
            table.append([dim, n, p, len(ms)] + [r2(np.median([m[k] for m in ms]))
                                                  for k in ("mean", "final", "unique", "head", "tail")])
        for p, ms in sel.items():
            if p == "uniform":
                continue
            row = [dim, n, p]
            for k in ("mean", "final"):
                pv, d = mwu([m[k] for m in ms], [m[k] for m in sel["uniform"]])
                row += [round(d, 3), pv]
            tests.append(row)
    for col in (4, 6):   # Holm over all tests of one metric
        for row, a in zip(tests, holm([row[col] for row in tests])):
            row.append(a)
    write_csv(f"{stem}_stats.csv", ["dim", "length", "profile", "runs", "mean_diversity", "final_diversity",
                                    "final_unique", "final_head_third", "final_tail_third"], table)
    write_csv(f"{stem}_tests.csv", ["dim", "length", "profile", "delta_mean", "p_mean", "delta_final",
                                    "p_final", "p_mean_holm", "p_final_holm"], tests)
    sig = [t for t in tests if t[7] < 0.05 or t[8] < 0.05]
    print(f"{len(tests)} tests vs uniform, {len(sig)} significant after Holm")
    print("written:", f"{stem.name}_stats.csv, {stem.name}_tests.csv")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seeds", type=int, default=10)
    r.add_argument("--seed-start", type=int, default=0)
    r.add_argument("--out", default="results/diversity.csv")
    r.add_argument("--lengths", nargs="+", type=int)
    r.add_argument("--dims", nargs="+", type=int, default=[2, 3])
    r.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("stats")
    s.add_argument("csv")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else stats(args.csv)


if __name__ == "__main__":
    main()
