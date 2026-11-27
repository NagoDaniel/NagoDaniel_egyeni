"""Run the GA for every mutation profile / sequence / dimension / seed and save one CSV row per run.

Run from the project root:
    .venv/bin/python src/run_experiment.py --seeds 10 --seed-start 1000 --out results/pilot.csv
Final experiment: --seeds 30 --seed-start 0 --out results/final.csv (pilot seeds are kept apart).
"""
import argparse
import csv
import time
from multiprocessing import Pool
from hp_core import BENCHMARKS_2D, BENCHMARKS_LARGE, MOVES_2D, MOVES_3D
from hp_ga import run_ga, PROFILES

# Target energy per sequence and dimension. 3D targets of the three short sequences are proven optima
# (src/exact_3d.c); the 36 and 48-mer targets are best-known values from the literature.
SEQS = list(BENCHMARKS_2D) + list(BENCHMARKS_LARGE)
TARGETS = {2: dict(BENCHMARKS_2D, **{s: t[0] for s, t in BENCHMARKS_LARGE.items()}),
           3: dict(zip(BENCHMARKS_2D, (-11, -13, -9)), **{s: t[1] for s, t in BENCHMARKS_LARGE.items()})}
ALPHABETS = {2: MOVES_2D, 3: MOVES_3D}
# Generation budget per sequence length (population stays 200). The two long sequences get more
# generations (pilot: results/budget_pilot.csv); equal budget for every profile on one sequence.
GENERATIONS = {20: 300, 24: 300, 25: 300, 36: 1500, 48: 1500}
FIELDS = ["profile", "dim", "seq", "seed", "success", "energy", "evaluations", "generations", "curve"]


def one_run(job):
    profile, dim, seq, seed, exp_base, generations = job
    r = run_ga(seq, ALPHABETS[dim], target_energy=TARGETS[dim][seq], mutation_profile=profile,
               generations=generations, seed=seed, exp_base=exp_base)
    return {"profile": profile, "dim": dim, "seq": seq, "seed": seed,
            "success": int(r["success"]), "energy": r["energy"],
            "evaluations": r["evaluations"], "generations": r["generations"],
            "curve": ";".join(str(fit) for fit, _ in r["history"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--out", default="results/pilot.csv")
    ap.add_argument("--profiles", nargs="+", default=list(PROFILES))
    ap.add_argument("--lengths", nargs="+", type=int, help="only these sequence lengths")
    ap.add_argument("--dims", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--exp-base", type=float, help="steepness of the exponential profiles")
    ap.add_argument("--long-generations", type=int, help="generation budget for the 36 and 48-mer")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (results identical)")
    args = ap.parse_args()

    seqs = [s for s in SEQS if not args.lengths or len(s) in args.lengths]
    budget = dict(GENERATIONS)
    if args.long_generations:
        budget.update({36: args.long_generations, 48: args.long_generations})
    jobs = [(p, d, s, seed, args.exp_base, budget[len(s)]) for d in args.dims for s in seqs for p in args.profiles
            for seed in range(args.seed_start, args.seed_start + args.seeds)]
    start = time.time()
    with open(args.out, "w", newline="") as f, Pool(args.workers) as pool:
        out = csv.DictWriter(f, FIELDS)
        out.writeheader()
        # imap keeps the job order, so the CSV is the same for any number of workers
        for k, row in enumerate(pool.imap(one_run, jobs, chunksize=1), 1):
            out.writerow(row)
            if k % 50 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} runs, {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
