"""Exploration: two simple ways to keep the population diverse. Run from the project root:

  .venv/bin/python src/hp_remedies.py run --seeds 30 --out results/remedies.csv --workers 9
  .venv/bin/python src/hp_remedies.py stats results/remedies.csv

Variants (all use the uniform mutation profile, population 200, same evaluation budget per run):
  base        the GA of the main experiment (mean mutation rate 1/(n-2), no immigrants)
  rate2/4/8   the mean mutation rate multiplied by 2, 4 or 8
  imm5/10/20  5, 10 or 20 % of every new generation are random new individuals (the elites are kept)
Immigrants are evaluated like children, so every variant costs 198 evaluations per generation.

Sequences: the 3D 24-mer and the 3D 36-mer, the same targets and generation budgets as the final
experiment. The runs stop at the target. The base rows must equal the uniform rows of final.csv.

CSV: one row per run: variant,dim,seq,seed,success,energy,evaluations,generations,div_mean,div_final
  div_mean / div_final = mean pairwise Hamming distance / length (see hp_ga.population_diversity),
  averaged over the run and at its last generation. Immigrants count as population members.

stats writes next to the CSV:
  <prefix>_stats.csv   per (length, variant): successes, ERT, energy median, diversity medians
  <prefix>_tests.csv   every variant against base (Fisher on successes, Mann-Whitney on evaluations
                       and energy), Holm over all comparisons of one metric
  <prefix>.tex         report table (best value of each column in bold)
"""
import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from hp_ga import run_ga
from hp_stats import mwu, fisher, holm, write_csv, r2, ert, evals, energies
from run_experiment import SEQS, TARGETS, ALPHABETS, GENERATIONS

DIM = 3
LENGTHS = (24, 36)
POP = 200
# variant -> (mutation rate factor, immigrants per generation)
VARIANTS = {"base": (1, 0), "rate2": (2, 0), "rate4": (4, 0), "rate8": (8, 0),
            "imm5": (1, POP // 20), "imm10": (1, POP // 10), "imm20": (1, POP // 5)}
FIELDS = ["variant", "dim", "seq", "seed", "success", "energy", "evaluations", "generations",
          "div_mean", "div_final"]


def one_run(job):
    variant, seq, seed = job
    factor, immigrants = VARIANTS[variant]
    r = run_ga(seq, ALPHABETS[DIM], pop_size=POP, generations=GENERATIONS[len(seq)],
               target_energy=TARGETS[DIM][seq], mutation_rate=factor / (len(seq) - 2), seed=seed,
               immigrants=immigrants, track_diversity=True)
    div = [h for h, _ in r["diversity"]]
    return {"variant": variant, "dim": DIM, "seq": seq, "seed": seed, "success": int(r["success"]),
            "energy": r["energy"], "evaluations": r["evaluations"], "generations": r["generations"],
            "div_mean": f"{np.mean(div):.4f}", "div_final": f"{div[-1]:.4f}"}


def run(args):
    seqs = [s for s in SEQS if len(s) in LENGTHS]
    jobs = [(v, s, seed) for s in seqs for v in VARIANTS
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
    return [{"variant": r["variant"], "n": len(r["seq"]), "seed": int(r["seed"]),
             "success": r["success"] == "1", "energy": int(r["energy"]) if r["energy"] else None,
             "evaluations": int(r["evaluations"]), "div_mean": float(r["div_mean"]),
             "div_final": float(r["div_final"])} for r in csv.DictReader(open(path))]


def cell(rows, n, variant):
    return [r for r in rows if r["n"] == n and r["variant"] == variant]


def stats(path):
    rows, stem = load(path), Path(path).with_suffix("")
    lengths = sorted({r["n"] for r in rows})
    variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]
    table, tests = [], []
    for n in lengths:
        base = cell(rows, n, "base")
        for v in variants:
            runs = cell(rows, n, v)
            e = energies(runs)
            table.append([n, v, len(runs), sum(r["success"] for r in runs), r2(ert(runs)),
                          r2(np.median(e)), r2(np.median([r["div_mean"] for r in runs])),
                          r2(np.median([r["div_final"] for r in runs]))])
            if v != "base":
                pe, de = mwu(evals(runs), evals(base))
                pg, dg = mwu(e, energies(base))
                tests.append([n, v, round(de, 3), pe, round(dg, 3), pg, fisher(runs, base)])
    for col in (3, 5, 6):
        for row, a in zip(tests, holm([row[col] for row in tests])):
            row.append(a)
    write_csv(f"{stem}_stats.csv", ["length", "variant", "runs", "successes", "ERT", "energy_median",
                                    "div_mean_median", "div_final_median"], table)
    write_csv(f"{stem}_tests.csv", ["length", "variant", "delta_evals", "p_evals", "delta_energy",
                                    "p_energy", "p_success", "p_evals_holm", "p_energy_holm",
                                    "p_success_holm"], tests)
    tex(table, f"{stem}.tex", lengths, variants)
    sig = [t for t in tests if min(t[7:]) < 0.05]
    print(f"{len(tests)} tests vs base, {len(sig)} significant after Holm")
    for t in sig:
        print("  ", t[0], t[1], "Holm p (evals, energy, success):", [round(x, 4) for x in t[7:]])


def tex(table, path, lengths, variants):
    """Report table: one row per variant, columns per sequence: successes, ERT (thousands),
    median energy, median run-average diversity. Best successes, ERT and energy are bold."""
    col = {(r[0], r[1]): r for r in table}
    best = {}
    for n in lengths:
        rs = [col[n, v] for v in variants]
        succ = [r[3] for r in rs]
        erts = [r[4] for r in rs if r[4] != ""]
        ens = [r[5] for r in rs]
        best[n] = (max(succ), min(erts) if erts else None, min(ens), max(ens))
    lines = ["\\begin{tabular}{l" + "rrrr" * len(lengths) + "}", "\\hline",
             "& " + " & ".join(f"\\multicolumn{{4}}{{c}}{{3D {n}}}" for n in lengths) + " \\\\",
             "változat & " + " & ".join("siker & ERT & energia & vált." for _ in lengths) + " \\\\", "\\hline"]
    for v in variants:
        cells = []
        for n in lengths:
            _, _, _, s, e, g, d, _ = col[n, v]
            b = best[n]
            ert_text = "nincs" if e == "" else f"{e / 1000:.0f}"
            en_text = f"${g:g}$"
            cells += [f"\\textbf{{{s}}}" if s == b[0] else str(s),
                      f"\\textbf{{{ert_text}}}" if e != "" and e == b[1] else ert_text,
                      f"\\textbf{{{en_text}}}" if g == b[2] and b[2] != b[3] else en_text,
                      f"{d:.2f}".replace(".", ",")]
        lines.append(f"\\texttt{{{v}}} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seeds", type=int, default=30)
    r.add_argument("--seed-start", type=int, default=0)
    r.add_argument("--out", default="results/remedies.csv")
    r.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("stats")
    s.add_argument("csv")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else stats(args.csv)


if __name__ == "__main__":
    main()
