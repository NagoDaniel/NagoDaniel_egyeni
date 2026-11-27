"""Statistics for an experiment CSV (no plots). Run from the project root:

  .venv/bin/python src/hp_stats.py results/final.csv

Writes next to the CSV (prefix = CSV name):
  <prefix>_stats.csv    per (dim, length, profile): runs, successes, success rate, energy and
                        evaluation median/IQR, ERT
  <prefix>_tests.csv    every profile against `uniform` in the same (dim, length) cell
  <prefix>_hyp.csv      the one-sided tests behind H1-H5, Holm-corrected within each hypothesis
  <prefix>_stats.tex    LaTeX table of the stats file
  <prefix>_{successes,ert}.tex
                        report tables: one row per profile, one column per (dim, length)

Conventions
  - evaluations: ALL runs. A failed run used its whole budget, so it counts with that value
    (a censored value: it can only make a slow profile look faster, never slower).
  - ERT = total evaluations of all runs / number of successful runs (blank if no run succeeded).
  - energy: final energy of the best fold. A run whose best fold is invalid has no energy and is
    left out of the energy tests (counted in the `no_energy` column).
  - tests: Mann-Whitney U (evaluations, energy), Fisher exact (success counts). delta is the
    rank-biserial effect size, positive = the first group has the larger values.
  - Holm correction over the family of tests of one hypothesis.
"""
import csv
import sys
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact
from hp_ga import PROFILES

ALPHA = 0.05
FIXED = ["uniform", "tail", "tail_exp", "head"]
ADAPTIVE_FAMILY = [p for p in PROFILES if p.startswith("adaptive")]


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({"profile": r["profile"], "dim": int(r["dim"]), "seq": r["seq"],
                     "n": len(r["seq"]), "seed": int(r["seed"]), "success": r["success"] == "1",
                     "energy": int(r["energy"]) if r["energy"] else None,
                     "evaluations": int(r["evaluations"])})
    return rows


def cell(rows, dim, n, profile):
    return [r for r in rows if r["dim"] == dim and r["n"] == n and r["profile"] == profile]


def energies(runs):
    return [r["energy"] for r in runs if r["energy"] is not None]


def evals(runs):
    return [r["evaluations"] for r in runs]


def ert(runs):
    ok = sum(r["success"] for r in runs)
    return sum(evals(runs)) / ok if ok else None


def mwu(a, b, alternative="two-sided"):
    """(p, delta) of Mann-Whitney a vs b. delta = 2U/(n1*n2) - 1 in [-1, 1]."""
    if not a or not b or len(set(a) | set(b)) == 1:
        return 1.0, 0.0
    u, p = mannwhitneyu(a, b, alternative=alternative)
    return float(p), 2 * u / (len(a) * len(b)) - 1


def fisher(a, b, alternative="two-sided"):
    """p of Fisher exact on success counts of runs a vs runs b."""
    sa, sb = sum(r["success"] for r in a), sum(r["success"] for r in b)
    return float(fisher_exact([[sa, len(a) - sa], [sb, len(b) - sb]], alternative=alternative)[1])


def holm(pvalues):
    """Holm step-down adjusted p-values, same order as the input."""
    order = np.argsort(pvalues)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def q(values):
    return tuple(np.percentile(values, [25, 50, 75])) if len(values) else (None,) * 3


def r2(x):
    return "" if x is None else round(float(x), 3)


def cells_of(rows):
    return sorted({(r["dim"], r["n"]) for r in rows})


def profiles_in(rows):
    return [p for p in PROFILES if any(r["profile"] == p for r in rows)]


def write_csv(path, header, data):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)


# ---------- tables ----------
def stats_table(rows):
    out = []
    for dim, n in cells_of(rows):
        for p in profiles_in(rows):
            runs = cell(rows, dim, n, p)
            if not runs:
                continue
            e, v = q(energies(runs)), q(evals(runs))
            ok = sum(r["success"] for r in runs)
            out.append([dim, n, p, len(runs), ok, round(ok / len(runs), 3), *map(r2, e), *map(r2, v),
                        r2(ert(runs)), len(runs) - len(energies(runs))])
    header = ["dim", "length", "profile", "runs", "successes", "success_rate",
              "energy_q1", "energy_median", "energy_q3", "evals_q1", "evals_median", "evals_q3",
              "ERT", "no_energy"]
    return header, out


def tests_vs_uniform(rows):
    out = []
    for dim, n in cells_of(rows):
        base = cell(rows, dim, n, "uniform")
        for p in profiles_in(rows):
            if p == "uniform":
                continue
            runs = cell(rows, dim, n, p)
            pe, de = mwu(evals(runs), evals(base))
            pg, dg = mwu(energies(runs), energies(base))
            out.append([dim, n, p, round(de, 3), pe, round(dg, 3), pg, fisher(runs, base)])
    for col in (4, 6, 7):  # Holm over all tests of one metric
        adj = holm([row[col] for row in out])
        for row, a in zip(out, adj):
            row.append(a)
    header = ["dim", "length", "profile", "delta_evals", "p_evals", "delta_energy", "p_energy",
              "p_success", "p_evals_holm", "p_energy_holm", "p_success_holm"]
    return header, out


def hypotheses(rows):
    """One-sided tests. Each row: hypothesis, dim, length, groups compared, metric, p, delta."""
    out = []
    profs = profiles_in(rows)
    for dim, n in cells_of(rows):
        base = cell(rows, dim, n, "uniform")
        # H1: tail-heavy fixed profiles need fewer evaluations than uniform
        for p in ("tail", "tail_exp"):
            if p in profs:
                pv, d = mwu(evals(cell(rows, dim, n, p)), evals(base), "less")
                out.append(["H1", dim, n, f"{p} < uniform", "evaluations", pv, d])
        # H2: the reverse profile needs more evaluations than uniform
        if "head" in profs:
            pv, d = mwu(evals(cell(rows, dim, n, "head")), evals(base), "greater")
            out.append(["H2", dim, n, "head > uniform", "evaluations", pv, d])
        # H3: an adaptive profile is not worse than the best fixed one (best = lowest ERT, else
        # lowest median evaluations). Tested as "adaptive needs MORE evaluations": a small p means worse.
        fixed = [p for p in FIXED if p in profs]
        best = min(fixed, key=lambda p: (ert(cell(rows, dim, n, p)) or float("inf"),
                                         np.median(evals(cell(rows, dim, n, p)))))
        for p in ADAPTIVE_FAMILY:
            if p in profs:
                pv, d = mwu(evals(cell(rows, dim, n, p)), evals(cell(rows, dim, n, best)), "greater")
                out.append(["H3", dim, n, f"{p} > best fixed ({best})", "evaluations", pv, d])
    # H4 / H5: 3D against 2D, same sequence, same profile
    for n in sorted({r["n"] for r in rows}):
        for p in profs:
            a2, a3 = cell(rows, 2, n, p), cell(rows, 3, n, p)
            if not a2 or not a3:
                continue
            pv, d = mwu(energies(a3), energies(a2), "less")
            out.append(["H4", "3v2", n, f"{p}: 3D < 2D", "energy", pv, d])
            pv, d = mwu(evals(a3), evals(a2), "greater")
            out.append(["H5", "3v2", n, f"{p}: 3D > 2D", "evaluations", pv, d])
    for h in sorted({row[0] for row in out}):
        idx = [i for i, row in enumerate(out) if row[0] == h]
        adj = holm([out[i][5] for i in idx])
        for i, a in zip(idx, adj):
            out[i].append(a)
    header = ["hypothesis", "dim", "length", "comparison", "metric", "p_one_sided", "delta", "p_holm"]
    return header, out


def hypothesis_summary(out):
    """Print how many tests per hypothesis are significant (raw and Holm)."""
    for h in sorted({row[0] for row in out}):
        rs = [row for row in out if row[0] == h]
        print(f"{h}: {sum(r[5] < ALPHA for r in rs)}/{len(rs)} raw p<{ALPHA}, "
              f"{sum(r[7] < ALPHA for r in rs)}/{len(rs)} after Holm")


def latex_table(header, data, path):
    keep = ["dim", "length", "profile", "runs", "successes", "energy_median", "evals_median", "ERT"]
    idx = [header.index(k) for k in keep]
    lines = ["\\begin{tabular}{rrlrrrrr}", "\\hline",
             " & ".join(k.replace("_", "\\_") for k in keep) + " \\\\", "\\hline"]
    for row in data:
        lines.append(" & ".join(str(row[i]).replace("_", "\\_") if row[i] != "" else "--" for i in idx) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")


def matrix_tex(rows, path, profiles, metric):
    """Report table: one row per profile, one column per (dim, length). The best value of each
    column is bold (most successes, lowest ERT; nothing is bold when no profile succeeded).
    metric "successes" = successes out of the runs, "ert" = ERT in thousands of evaluations
    ("nincs" when no run succeeded)."""
    cells = cells_of(rows)
    values = {}
    for dim, n in cells:
        for p in profiles:
            runs = cell(rows, dim, n, p)
            values[p, dim, n] = sum(r["success"] for r in runs) if metric == "successes" else ert(runs)
    lines = ["\\begin{tabular}{l" + "r" * len(cells) + "}", "\\hline",
             "profil & " + " & ".join(f"{d}D {n}" for d, n in cells) + " \\\\", "\\hline"]
    best = {}
    for dim, n in cells:
        col = [values[p, dim, n] for p in profiles if values[p, dim, n]]
        best[dim, n] = (max(col) if metric == "successes" else min(col)) if col else None
    for p in profiles:
        out = []
        for dim, n in cells:
            v = values[p, dim, n]
            text = ("nincs" if not v else f"{v / 1000:.0f}") if metric == "ert" else str(v)
            out.append(f"\\textbf{{{text}}}" if v and v == best[dim, n] else text)
        lines.append(p.replace("_", "\\_") + " & " + " & ".join(out) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    path = Path(sys.argv[1])
    rows, stem = load(path), path.with_suffix("")
    header, data = stats_table(rows)
    write_csv(f"{stem}_stats.csv", header, data)
    latex_table(header, data, f"{stem}_stats.tex")
    for metric in ("successes", "ert"):
        matrix_tex(rows, f"{stem}_{metric}.tex", profiles_in(rows), metric)
    header, data = tests_vs_uniform(rows)
    write_csv(f"{stem}_tests.csv", header, data)
    header, hyp = hypotheses(rows)
    write_csv(f"{stem}_hyp.csv", header, hyp)
    hypothesis_summary(hyp)
    print("written:", ", ".join(f"{stem.name}_{s}" for s in ("stats.csv", "stats.tex", "tests.csv", "hyp.csv")))


if __name__ == "__main__":
    main()
