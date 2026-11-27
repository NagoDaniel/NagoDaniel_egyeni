"""Genetic algorithm for HP folding (2D and 3D). Run: .venv/bin/python hp_ga.py

Individual = string of relative moves. Settings:
  - tournament selection (size 3): simple, steady pressure, no fitness scaling needed
  - one-point crossover: keeps runs of moves (local fold pieces) together
  - per-move mutation, mean rate 1/length (about one changed move per child). The rate can
    depend on the position in the move string (see PROFILES). The mean stays the same,
    only the shape changes, so profiles are compared fairly.
  - elitism (2 best copied unchanged): best fold is never lost
  - optional random immigrants: every generation some children are replaced by fresh random
    individuals (never the elites). The population size and the evaluations per generation stay
    the same, so runs with and without immigrants use the same budget.
  - collisions allowed but penalised (5 each), so the GA can pass through invalid folds;
    one overlap can fake at most 4 contacts (6 neighbours - 2 chain bonds), so 5 > 4
"""
import random
from hp_core import fitness, energy, MOVES_2D, MOVES_3D, BENCHMARKS_2D

# Position weights for mutation. Move i changes the fold from residue i+3 to the end,
# so an early move is a big change and a late move is a small one. b = exponential base.
PROFILES = {
    "uniform": lambda i, n, b: 1,
    "tail": lambda i, n, b: i + 1,          # late moves mutate more (linear)
    "tail_exp": lambda i, n, b: b ** i,     # late moves mutate much more (exponential)
    "head": lambda i, n, b: n - i,          # control: early moves mutate more
    # Adaptive: the shape changes over the run (see adaptive_rates). Weights = start shape.
    "adaptive": lambda i, n, b: i + 1,          # tail -> uniform
    "adaptive_rev": lambda i, n, b: 1,          # uniform -> tail
    "adaptive_exp": lambda i, n, b: b ** i,     # tail_exp -> uniform
    "adaptive_exp_rev": lambda i, n, b: 1,      # uniform -> tail_exp
}
# profile -> (shaped profile it blends with uniform, True if it starts uniform)
ADAPTIVE = {"adaptive": ("tail", False), "adaptive_rev": ("tail", True),
            "adaptive_exp": ("tail_exp", False), "adaptive_exp_rev": ("tail_exp", True)}
EXP_BASE = 1.2   # steepness of the exponential profiles (chosen on pilot seeds, see PLAN.md)


def position_rates(length, profile, mean_rate, exp_base=None):
    """Mutation probability for each move position. Their mean is mean_rate."""
    if profile == "uniform":
        return [mean_rate] * length
    base = EXP_BASE if exp_base is None else exp_base
    weights = [PROFILES[profile](i, length, base) for i in range(length)]
    scale = mean_rate * length / sum(weights)
    return [min(1.0, w * scale) for w in weights]


def adaptive_rates(length, mean_rate, progress, shape="tail", reverse=False, exp_base=None):
    """Blend between `shape` and uniform. progress 0 -> shape, 1 -> uniform; reverse flips it
    (starts uniform, ends in `shape`). A mix of two profiles with the same mean keeps that mean."""
    if reverse:
        progress = 1 - progress
    shaped = position_rates(length, shape, mean_rate, exp_base)
    return [(1 - progress) * t + progress * mean_rate for t in shaped]


def population_diversity(moves_list):
    """Genotype diversity of a population of move strings, no random numbers used.
    Returns (hamming, unique, per_position):
      hamming       mean pairwise Hamming distance divided by the string length (0 = all equal)
      unique        share of distinct strings (1/N = all equal, 1 = all different)
      per_position  the same distance measured at each move position separately
    The pairwise mean is computed from letter counts per position, not by comparing all pairs:
    differing pairs at a position = (N^2 - sum of squared counts) / 2."""
    n, length = len(moves_list), len(moves_list[0])
    per_position = []
    for i in range(length):
        counts = {}
        for m in moves_list:
            counts[m[i]] = counts.get(m[i], 0) + 1
        per_position.append((n * n - sum(c * c for c in counts.values())) / (n * (n - 1)))
    return sum(per_position) / length, len(set(moves_list)) / n, per_position


def run_ga(seq, alphabet, pop_size=200, generations=300, tournament_size=3,
           crossover_rate=0.9, mutation_rate=None, elite_count=2,
           collision_penalty=5, target_energy=None, mutation_profile="uniform", seed=None,
           exp_base=None, track_diversity=False, immigrants=0):
    """Returns a dict with the best fold, its energy, history, evaluations and generations used.

    Stops early once a valid fold with energy <= target_energy is found.
    immigrants = number of random new individuals per generation (default 0 = off, run unchanged).
    track_diversity adds "diversity" (hamming, unique) of the population for generation 0, 1, ...
    and "positions" (per-position distance of the last population). It does not change the run.
    """
    rng = random.Random(seed)
    length = len(seq) - 2
    if mutation_rate is None:
        mutation_rate = 1 / length
    rates = position_rates(length, mutation_profile, mutation_rate, exp_base)

    def score(moves):
        return fitness(seq, moves, collision_penalty)

    def pick_parent(scored):
        return min(rng.sample(scored, tournament_size))[1]

    def crossover(a, b):
        if rng.random() > crossover_rate:
            return a
        cut = rng.randint(1, length - 1)
        return a[:cut] + b[cut:]

    def mutate(moves):
        return "".join(rng.choice(alphabet.replace(m, "")) if rng.random() < rates[i] else m
                       for i, m in enumerate(moves))

    population = ["".join(rng.choices(alphabet, k=length)) for _ in range(pop_size)]
    scored = sorted((score(m), m) for m in population)
    evaluations = pop_size
    history = [scored[0]]  # best (fitness, moves) of each generation
    diversity = [population_diversity([m for _, m in scored])[:2]] if track_diversity else []

    for gen in range(generations):
        if mutation_profile in ADAPTIVE:   # progress = share of the generation budget used
            shape, reverse = ADAPTIVE[mutation_profile]
            rates[:] = adaptive_rates(length, mutation_rate, gen / max(1, generations - 1),
                                      shape, reverse, exp_base)
        best_fit, best_moves = scored[0]
        if target_energy is not None and energy(seq, best_moves) is not None \
                and best_fit <= target_energy:
            break
        children = [mutate(crossover(pick_parent(scored), pick_parent(scored)))
                    for _ in range(pop_size - elite_count - immigrants)]
        children += ["".join(rng.choices(alphabet, k=length)) for _ in range(immigrants)]
        scored = sorted(scored[:elite_count] + [(score(c), c) for c in children])
        evaluations += len(children)
        history.append(scored[0])
        if track_diversity:
            diversity.append(population_diversity([m for _, m in scored])[:2])

    best_fit, best_moves = scored[0]
    best_energy = energy(seq, best_moves)
    success = target_energy is not None and best_energy is not None and best_fit <= target_energy
    result = {"moves": best_moves, "energy": best_energy, "fitness": best_fit,
              "history": history, "evaluations": evaluations,
              "generations": len(history) - 1, "success": success}
    if track_diversity:
        result["diversity"] = diversity
        result["positions"] = population_diversity([m for _, m in scored])[2]
    return result


def run():
    for seq, known_2d in BENCHMARKS_2D.items():
        for dim, alphabet in [(2, MOVES_2D), (3, MOVES_3D)]:
            r = run_ga(seq, alphabet, seed=0)
            known = f"(known 2D best {known_2d})" if dim == 2 else ""
            print(f"{dim}D n={len(seq)}  energy {r['energy']} {known}  "
                    f"evals {r['evaluations']}  moves {r['moves']}")
        return
if __name__ == "__main__":
    run()
