"""HP model helpers: decode a fold, count contacts and collisions, score it.

A fold is a string of n-2 relative moves:
    E = forward, B = left, J = right, F = up, L = down   (2D uses only E, B, J)
Residue 1 sits at the origin, residue 2 at +x (first bond fixed).
"""
from itertools import product

MOVES_2D = "EBJ"
MOVES_3D = "EBJFL"
NEIGHBOURS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

# Benchmark sequences with known best 2D energies.
BENCHMARKS_2D = {
    "HPHPPHHPHPPHPHHPPHPH": -9,        # 20
    "HHPPHPPHPPHPPHPPHPPHPPHH": -9,    # 24
    "PPHPPHHPPPPHHPPPPHHPPPPHH": -8,   # 25
}

# Larger benchmarks, (2D best known, 3D best known). 2D: standard literature values.
# 3D: arXiv 2105.13226 Table 1 (36-mer: paper's expansion has an extra trailing P, dropped here).
# Best known, NOT proven optimal here: see results/benchmarks.csv.
BENCHMARKS_LARGE = {
    "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP": (-14, -18),                    # 36
    "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH": (-23, -31),        # 48
}


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def neg(a):
    return (-a[0], -a[1], -a[2])


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def decode(moves):
    """Relative moves -> list of (x, y, z) positions. Overlaps are allowed here."""
    heading, up = (1, 0, 0), (0, 0, 1)
    positions = [(0, 0, 0), (1, 0, 0)]
    for move in moves:
        if move == "B":
            heading = cross(up, heading)
        elif move == "J":
            heading = neg(cross(up, heading))
        elif move == "F":
            heading, up = up, neg(heading)
        elif move == "L":
            heading, up = neg(up), heading
        positions.append(add(positions[-1], heading))
    return positions


def count_collisions(positions):
    """Number of residues sitting on an already occupied point."""
    return len(positions) - len(set(positions))


def contact_pairs(seq, positions):
    """H-H pairs (i, j) adjacent on the lattice but not neighbours in the chain."""
    where = {}
    for i, p in enumerate(positions):
        where.setdefault(p, []).append(i)
    pairs = set()
    for i, p in enumerate(positions):
        if seq[i] != "H":
            continue
        for step in NEIGHBOURS:
            for j in where.get(add(p, step), []):
                if j > i + 1 and seq[j] == "H":
                    pairs.add((i, j))
    return sorted(pairs)


def fitness(seq, moves, collision_penalty=5):
    """Lower is better: -contacts + penalty * collisions. Valid fold => fitness = energy."""
    positions = decode(moves)
    return -len(contact_pairs(seq, positions)) + collision_penalty * count_collisions(positions)


def energy(seq, moves):
    """Energy of a valid fold (None if the chain overlaps itself)."""
    positions = decode(moves)
    if count_collisions(positions):
        return None
    return -len(contact_pairs(seq, positions))


def brute_force(seq, alphabet):
    """Exact best fold by trying all moves strings. Only for short sequences."""
    best_energy, best_moves = 0, None
    for moves in product(alphabet, repeat=len(seq) - 2):
        e = energy(seq, moves)
        if e is not None and (best_moves is None or e < best_energy):
            best_energy, best_moves = e, "".join(moves)
    return best_energy, best_moves
