"""Plain tests, hand-checked small cases. Run: .venv/bin/python test_hp.py"""
from hp_core import (decode, count_collisions, contact_pairs, fitness, energy,
                     brute_force, MOVES_2D, MOVES_3D)
from hp_ga import run_ga, position_rates, adaptive_rates, PROFILES, ADAPTIVE


def test_straight_line():
    assert decode("EE") == [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]


def test_left_and_right_turns():
    assert decode("B")[-1] == (1, 1, 0)
    assert decode("J")[-1] == (1, -1, 0)
    # after a left turn, forward keeps going in +y
    assert decode("BE")[-1] == (1, 2, 0)


def test_up_and_down():
    assert decode("F")[-1] == (1, 0, 1)
    assert decode("L")[-1] == (1, 0, -1)
    # up then forward: keeps climbing
    assert decode("FE")[-1] == (1, 0, 2)
    # up twice: heading flips to -x, so we come back over the start
    assert decode("FF")[-1] == (0, 0, 1)


def test_left_after_up_uses_new_frame():
    # after F: heading = +z, up = -x, so left = up x heading = -x x +z = +y
    assert decode("FB")[-1] == (1, 1, 1)


def test_2d_moves_stay_flat():
    assert all(z == 0 for _, _, z in decode("BJEBBJ"))


def test_square_has_one_contact():
    # HHHH folded into a square: residue 1 touches residue 4
    pos = decode("BB")
    assert pos == [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    assert contact_pairs("HHHH", pos) == [(0, 3)]
    assert contact_pairs("HHHP", pos) == []
    assert energy("HHHH", "BB") == -1


def test_chain_neighbours_are_not_contacts():
    assert contact_pairs("HHHH", decode("EE")) == []


def test_collision_is_counted_and_penalised():
    # five residues around a square: the fifth lands back on the origin
    pos = decode("BBB")
    assert pos[4] == (0, 0, 0)
    assert count_collisions(pos) == 1
    assert energy("HHHHH", "BBB") is None
    assert fitness("HHHHH", "BBB", collision_penalty=2) == -len(contact_pairs("HHHHH", pos)) + 2


def test_valid_fold_fitness_equals_energy():
    assert fitness("HHHH", "BB") == energy("HHHH", "BB") == -1


def test_brute_force_small():
    # 2x2x2 cube for 8 H residues: 12 cube edges - 7 chain bonds = 5 contacts
    e3, _ = brute_force("HHHHHHHH", MOVES_3D)
    assert e3 == -5
    # in 2D the best 8-mer of H is a 2x4 / U shape: 10 lattice edges max - 7 bonds = 3
    e2, _ = brute_force("HHHHHHHH", MOVES_2D)
    assert e2 == -3


def test_3d_never_worse_than_2d():
    seq = "HHPHHPHHHH"
    assert brute_force(seq, MOVES_3D)[0] <= brute_force(seq, MOVES_2D)[0]


def test_ga_finds_small_optimum():
    seq = "HHPHHPHHHH"
    for alphabet in (MOVES_2D, MOVES_3D):
        best, _ = brute_force(seq, alphabet)
        result = run_ga(seq, alphabet, generations=100, seed=1)
        assert result["energy"] == best


def test_profiles_keep_mean_rate():
    for profile in PROFILES:
        rates = position_rates(18, profile, 1 / 18)
        assert abs(sum(rates) / 18 - 1 / 18) < 1e-12


def test_profile_shapes():
    for shape in ("tail", "tail_exp"):
        r = position_rates(10, shape, 0.1)
        assert r == sorted(r) and r[0] < r[-1]
    head = position_rates(10, "head", 0.1)
    assert head == sorted(head, reverse=True) and head[0] > head[-1]
    assert position_rates(10, "uniform", 0.1) == [0.1] * 10


def test_exp_base_steepness():
    flat, steep = position_rates(18, "tail_exp", 1 / 18, 1.1), position_rates(18, "tail_exp", 1 / 18, 1.3)
    assert steep[-1] > flat[-1] and steep[0] < flat[0]
    assert abs(sum(steep) / 18 - 1 / 18) < 1e-12


def test_adaptive_rates():
    for shape in ("tail", "tail_exp"):
        shaped = position_rates(18, shape, 1 / 18)
        assert adaptive_rates(18, 1 / 18, 0, shape) == shaped
        assert all(abs(r - 1 / 18) < 1e-12 for r in adaptive_rates(18, 1 / 18, 1, shape))
        # reverse: starts uniform, ends in the shape
        assert all(abs(r - 1 / 18) < 1e-12 for r in adaptive_rates(18, 1 / 18, 0, shape, True))
        assert adaptive_rates(18, 1 / 18, 1, shape, True) == shaped
        for progress in (0, 0.3, 0.7, 1):
            for rev in (False, True):
                assert abs(sum(adaptive_rates(18, 1 / 18, progress, shape, rev)) / 18 - 1 / 18) < 1e-12


def test_adaptive_profiles_start_as_documented():
    for name, (shape, reverse) in ADAPTIVE.items():
        start = position_rates(18, name, 1 / 18)
        want = [1 / 18] * 18 if reverse else position_rates(18, shape, 1 / 18)
        assert all(abs(a - b) < 1e-12 for a, b in zip(start, want))


def test_uniform_profile_unchanged():
    # moves found before profiles were added (same seed => same run)
    seq = "HPHPPHHPHPPHPHHPPHPH"
    assert run_ga(seq, MOVES_2D, seed=0)["moves"] == "JEJJBBJBJJEJBBJJEJ"
    assert run_ga(seq, MOVES_3D, seed=0)["moves"] == "BEBBJFFEBBEFEEBBFB"


def test_ga_reports_generations_and_success():
    seq = "HHPHHPHHHH"
    r = run_ga(seq, MOVES_2D, generations=100, target_energy=-4, seed=1)
    assert r["success"] and r["generations"] == len(r["history"]) - 1
    assert run_ga(seq, MOVES_2D, generations=2, seed=1)["success"] is False


def test_fold_animation_kinematics():
    # joint interpolation gives the decoder's fold at both ends, bonds stay length 1 in between
    import random
    import numpy as np
    from hp_fold3d import chain
    rng = random.Random(3)
    for _ in range(50):
        a = "".join(rng.choices(MOVES_3D, k=12))
        b = "".join(rng.choices(MOVES_3D, k=12))
        assert np.allclose(chain(a, b, 0), decode(a)) and np.allclose(chain(a, b, 1), decode(b))
        for t in (0.25, 0.5, 0.8):
            bonds = np.linalg.norm(np.diff(chain(a, b, t), axis=0), axis=1)
            assert np.allclose(bonds, 1)


def test_holm_and_mwu():
    from hp_stats import holm, mwu
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]   # step-down, monotone
    assert holm([0.5]) == [0.5]
    p, d = mwu([1, 1, 1], [1, 1, 1])
    assert p == 1.0 and d == 0.0                             # all ties: no difference
    p, d = mwu([1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15], "less")
    assert p < 0.01 and d == -1.0                            # first group clearly smaller


def test_population_diversity():
    from hp_ga import population_diversity
    h, u, pos = population_diversity(["EBJ"] * 10)
    assert h == 0 and u == 0.1 and pos == [0, 0, 0]          # identical population
    h, u, pos = population_diversity(["EB", "BE"])           # two strings differing everywhere
    assert h == 1 and u == 1 and pos == [1, 1]
    h, u, pos = population_diversity(["EE", "EE", "EB", "EB"])   # only the 2nd position varies
    assert pos[0] == 0 and abs(pos[1] - 4 / 6) < 1e-12           # 4 of the 6 pairs differ there
    assert abs(h - 1 / 3) < 1e-12 and u == 0.5


def test_diversity_tracking_does_not_change_the_run():
    from hp_ga import run_ga
    kw = dict(pop_size=30, generations=12, seed=4, mutation_profile="adaptive")
    a, b = run_ga("HPHPPHHPHPPHPHHPPHPH", MOVES_3D, **kw), run_ga("HPHPPHHPHPPHPHHPPHPH", MOVES_3D, track_diversity=True, **kw)
    assert a["moves"] == b["moves"] and a["history"] == b["history"]
    assert len(b["diversity"]) == 13 and len(b["positions"]) == 18   # generation 0..12, one value per move


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
