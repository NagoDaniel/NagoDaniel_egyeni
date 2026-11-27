"""3D movie of a GA run: the chain is drawn as solid geometry and folds smoothly from one
best fold to the next. Run from the project root:

  .venv/bin/python src/hp_fold3d.py run SEQ [--dim 3] [--seed 0] [--profile uniform] [--out F.json]
  .venv/bin/python src/hp_fold3d.py render F.json [--out videos/ga_fold3d.mp4] [--fps 30]
  .venv/bin/python src/hp_fold3d.py videos [--dims 3] [--seed 0] [--profile uniform] [--lengths 20 48]

`run` saves the GA output, `render` turns it into a movie (.mp4 needs ffmpeg, .gif uses pillow).
Input JSON (one entry per generation = best fold of that generation):
  {"seq": "HPHP...", "dim": 3, "seed": 0, "profile": "uniform",
   "history": [[fitness, "EBJF..."], ...]}

How the motion works: moves are relative, so changing move i rotates the whole chain after
residue i+2 around that residue, like a joint. Between two best folds every joint turns from its
old move to its new move at the same time (quaternion interpolation), and the positions are
rebuilt from the joints. Bonds keep length 1 the whole time, so the chain really folds.
A turn by 180 degrees (left <-> right, up <-> down) goes through "forward", never backwards.

Look: H = red sphere, P = blue cube, bond = grey tube, H-H contact = gold tube (fades in as two
H come close), overlapping residues (collisions) turn magenta.
"""
import argparse
import json
import shutil
import textwrap
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from hp_core import decode, contact_pairs, count_collisions, MOVES_2D, MOVES_3D
from hp_ga import run_ga

ROOT = Path(__file__).resolve().parent.parent

# ---------- kinematics ----------
# Frame = matrix with columns (heading, left, up). A move multiplies the frame from the right
# by a fixed local rotation (same rules as hp_core.decode).
_e1, _e2, _e3 = np.eye(3)
LOCAL = {
    "E": np.eye(3),
    "B": np.column_stack([_e2, -_e1, _e3]),   # heading <- left
    "J": np.column_stack([-_e2, _e1, _e3]),   # heading <- right
    "F": np.column_stack([_e3, _e2, -_e1]),   # heading <- up, up <- -heading
    "L": np.column_stack([-_e3, _e2, _e1]),   # heading <- down, up <- heading
}
OPPOSITE = {("B", "J"), ("J", "B"), ("F", "L"), ("L", "F")}


def axis_angle_matrix(axis, angle):
    """Rodrigues: rotation by angle around a unit axis."""
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def partial_rotation(R, t):
    """Rotation R scaled to fraction t of its angle (R is never a 180 degree turn here)."""
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (4 * w)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
    return axis_angle_matrix(v / s, 2 * np.arctan2(s, w) * t)


def joint(a, b, t):
    """Local rotation of a joint going from move a (t=0) to move b (t=1)."""
    if a == b or t <= 0:
        return LOCAL[a]
    if t >= 1:
        return LOCAL[b]
    if (a, b) in OPPOSITE:  # half way through "forward"
        return joint(a, "E", 2 * t) if t < 0.5 else joint("E", b, 2 * t - 1)
    A = LOCAL[a]
    return A @ partial_rotation(A.T @ LOCAL[b], t)


def chain(moves_a, moves_b, t):
    """Residue positions (n x 3) of the chain part way (t in 0..1) from moves_a to moves_b."""
    frame = np.eye(3)
    pos = [np.zeros(3), np.array([1.0, 0, 0])]
    for a, b in zip(moves_a, moves_b):
        frame = frame @ joint(a, b, t)
        pos.append(pos[-1] + frame[:, 0])
    return np.array(pos)


# ---------- meshes (lists of polygons + outward normals) ----------
def _sphere(n_lat=9, n_lon=14):
    th = np.linspace(0, np.pi, n_lat + 1)
    ph = np.linspace(0, 2 * np.pi, n_lon + 1)
    pt = lambda i, j: np.array([np.sin(th[i]) * np.cos(ph[j]), np.sin(th[i]) * np.sin(ph[j]),
                                np.cos(th[i])])
    quads = [[pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)]
             for i in range(n_lat) for j in range(n_lon)]
    quads = np.array(quads)
    normals = quads.mean(1)
    return quads, normals / np.linalg.norm(normals, axis=1, keepdims=True)


def _cube():
    quads, normals = [], []
    for axis in range(3):
        for sign in (-1, 1):
            u, v = [k for k in range(3) if k != axis]
            face = []
            for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                p = np.zeros(3)
                p[axis], p[u], p[v] = sign, a, b
                face.append(p)
            n = np.zeros(3)
            n[axis] = sign
            quads.append(face)
            normals.append(n)
    return np.array(quads), np.array(normals)


SPHERE, CUBE = _sphere(), _cube()
H_RADIUS, P_HALF = 0.34, 0.25
BOND_R, CONTACT_R = 0.09, 0.06
SIDES = 12
_ang = np.linspace(0, 2 * np.pi, SIDES + 1)


def tube(p, q, r):
    """Open cylinder from p to q: (SIDES, 4, 3) quads and their normals."""
    d = q - p
    d = d / np.linalg.norm(d)
    helper = _e3 if abs(d[2]) < 0.9 else _e1
    u = np.cross(d, helper)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    ring = np.outer(np.cos(_ang), u) + np.outer(np.sin(_ang), v)  # (SIDES+1, 3)
    quads = np.stack([p + r * ring[:-1], q + r * ring[:-1], q + r * ring[1:], p + r * ring[1:]], 1)
    normals = (ring[:-1] + ring[1:]) / 2
    return quads, normals / np.linalg.norm(normals, axis=1, keepdims=True)


# ---------- colours and lighting ----------
def rgb(hex_color):
    return np.array([int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)])


H_RGB, P_RGB = rgb("#d6402f"), rgb("#2f6fd6")
BOND_RGB, CONTACT_RGB, CLASH_RGB = rgb("#9a9a9a"), rgb("#f2b705"), rgb("#d633c4")
BG = "#14161c"


def shade(base, normals, eye, alpha):
    """Blinn-Phong with a light a bit above and left of the camera."""
    light = eye + np.array([0, 0, 0.8]) + 0.5 * np.cross(np.array([0, 0, 1.0]), eye)
    light /= np.linalg.norm(light)
    half = (light + eye) / np.linalg.norm(light + eye)
    diff = np.clip(normals @ light, 0, 1)[:, None]
    spec = np.clip(normals @ half, 0, 1)[:, None] ** 40
    col = np.clip(base * (0.28 + 0.72 * diff) + 0.45 * spec, 0, 1)
    return np.column_stack([col, alpha])


def scene(seq, pos, eye):
    """All polygons and their RGBA colours for one frame."""
    polys, normals, bases, alphas = [], [], [], []

    def add(q, n, color, a=1.0):
        polys.append(q)
        normals.append(n)
        bases.append(np.broadcast_to(color, n.shape))
        alphas.append(np.full(len(n), a))

    n = len(pos)
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    clash = ((dist < 0.5) & ~np.eye(n, dtype=bool)).any(1)

    for i in range(n - 1):
        add(*tube(pos[i], pos[i + 1], BOND_R), BOND_RGB)
    h = [i for i in range(n) if seq[i] == "H"]
    for i in h:
        for j in h:
            if j > i + 1 and 0.5 <= dist[i, j] < 1.4:  # fades in as the pair closes in
                fade = float(np.clip((1.4 - dist[i, j]) / 0.35, 0, 1))
                add(*tube(pos[i], pos[j], CONTACT_R), CONTACT_RGB, 0.9 * fade)
    for i in range(n):
        if seq[i] == "H":
            q, nn = SPHERE
            add(pos[i] + H_RADIUS * q, nn, CLASH_RGB if clash[i] else H_RGB)
        else:
            q, nn = CUBE
            add(pos[i] + P_HALF * q, nn, CLASH_RGB if clash[i] else P_RGB)

    normals = np.concatenate(normals)
    colors = shade(np.concatenate(bases), normals, eye, np.concatenate(alphas))
    return np.concatenate(polys), colors


# ---------- timeline ----------
def smoothstep(t):
    return t * t * (3 - 2 * t)


def keyframes(history):
    """Generation indices where the best fitness improved (plus generation 0), and the last
    generation if its best fold differs (same fitness, other moves: ties are broken by moves)."""
    keys = [g for g in range(len(history)) if g == 0 or history[g][0] < history[g - 1][0]]
    if history[-1][1] != history[keys[-1]][1]:
        keys.append(len(history) - 1)
    return keys


def timeline(history, move_frames, hold_frames, end_frames):
    """One dict per movie frame: the two folds, the blend t, and the generation shown."""
    keys = keyframes(history)
    frames = []
    for k, g in enumerate(keys):
        moves = history[g][1]
        hold = end_frames if k == len(keys) - 1 else hold_frames
        frames += [dict(a=moves, b=moves, t=0.0, gen=g, key=g)] * hold
        if k + 1 < len(keys):
            g2 = keys[k + 1]
            for f in range(1, move_frames + 1):
                t = smoothstep(f / move_frames)
                frames.append(dict(a=moves, b=history[g2][1], t=t,
                                   gen=g + (g2 - g) * t, key=g2 if t > 0.5 else g))
    return frames


def smooth(values, strength=0.92):
    """Forward and backward exponential smoothing (no lag) of a per-frame series."""
    out = np.array(values, float)
    for order in (range(1, len(out)), range(len(out) - 2, -1, -1)):
        for i in order:
            j = i - 1 if order.step == 1 else i + 1
            out[i] = strength * out[j] + (1 - strength) * out[i]
    return out


# ---------- rendering ----------
CAMERA_SMOOTH = 0.97          # 0 = camera glued to the chain centre, 1 = frozen
ZOOM_SMOOTH = 0.92            # same for the zoom (lower = follows the chain size faster)
SPIN_DEG_PER_FRAME = 0.15     # slow orbit around the chain
def render(run, out, fps=30, move_frames=24, hold_frames=14, end_frames=120, dpi=100):
    seq, history = run["seq"], [(f, m) for f, m in run["history"]]
    frames = timeline(history, move_frames, hold_frames, end_frames)
    all_pos = [chain(fr["a"], fr["b"], fr["t"]) for fr in frames]
    # camera: follows the centre, zooms out early enough to keep the whole chain in view
    centers = smooth([p.mean(0) for p in all_pos], CAMERA_SMOOTH)
    reach = [np.linalg.norm(p - c, axis=1).max() + 0.8 for p, c in zip(all_pos, centers)]
    w = 20   # look about +-0.7 s ahead and behind, so the zoom eases instead of jumping at each new fold
    radius = smooth([max(reach[max(0, i - w):i + w + 1]) for i in range(len(reach))], ZOOM_SMOOTH)
    gens = np.arange(len(history))
    fits = np.array([f for f, _ in history])

    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    ax = fig.add_axes([0.0, 0.08, 0.64, 0.84], projection="3d", facecolor=BG)
    ax.computed_zorder = False
    side = fig.add_axes([0.69, 0.42, 0.28, 0.38], facecolor=BG)
    side.step(gens, fits, where="post", color="#c8c8c8", lw=1.5)
    for k in keyframes(history):
        side.plot(k, history[k][0], "o", ms=3, color="#777")
    marker, = side.plot([], [], "o", ms=9, color=CONTACT_RGB, zorder=5)
    optimum, note = run.get("optimum"), run.get("optimum_note", "")
    if optimum is not None:
        side.axhline(optimum, color=CONTACT_RGB, lw=1, ls=(0, (4, 3)), zorder=2)
        side.text(len(history) - 1 if len(history) > 1 else 1, optimum, f"{note} {optimum}  ", color=CONTACT_RGB,
                  fontsize=8, ha="right", va="bottom")
        side.set_ylim(min(optimum, fits.min()) - 1, fits.max() + 1)
    side.set_xlabel("generation", color="#bbb")
    side.set_ylabel("best fitness  (-contacts + 5 x collisions)", color="#bbb", fontsize=9)
    side.tick_params(colors="#999")
    for s in side.spines.values():
        s.set_color("#555")
    side.set_title("GA progress", color="#ddd", fontsize=11)

    title = f"GA folding an HP chain on the {run['dim']}D lattice   n={len(seq)}   seed {run.get('seed')}"
    fig.text(0.03, 0.95, title, color="#eee", fontsize=15, weight="bold")
    fig.text(0.03, 0.915, seq, color="#999", fontsize=11, family="monospace")
    info = fig.text(0.69, 0.31, "", color="#eee", fontsize=12, family="monospace", va="top")
    legend = [Line2D([], [], marker="o", ls="", ms=12, color=H_RGB, label="H  hydrophobic (sphere)"),
              Line2D([], [], marker="s", ls="", ms=10, color=P_RGB, label="P  polar (cube)"),
              Line2D([], [], color=BOND_RGB, lw=4, label="chain bond"),
              Line2D([], [], color=CONTACT_RGB, lw=3, label="H-H contact (energy -1)"),
              Line2D([], [], marker="o", ls="", ms=10, color=CLASH_RGB, label="collision (overlap)")]
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.02, 0.01), frameon=False,
               labelcolor="#ddd", fontsize=10, ncol=5, handlelength=1.6)

    def draw(f):
        fr, pos = frames[f], all_pos[f]
        c, r = centers[f], radius[f]
        azim, elev = 35 + SPIN_DEG_PER_FRAME * f, 22 + 3 * np.sin(f / 200)
        a, e = np.radians(azim), np.radians(elev)
        eye = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])

        ax.clear()
        ax.set_axis_off()
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r); ax.set_box_aspect((1, 1, 1), zoom=1.3)
        ax.view_init(elev, azim)

        # floor: lattice grid and the chain's shadow
        z0 = c[2] - r
        g0, g1 = np.floor(c[:2] - r), np.ceil(c[:2] + r)
        grid = [[(x, g0[1], z0), (x, g1[1], z0)] for x in np.arange(g0[0], g1[0] + 1)]
        grid += [[(g0[0], y, z0), (g1[0], y, z0)] for y in np.arange(g0[1], g1[1] + 1)]
        ax.add_collection3d(Line3DCollection(grid, colors="#2a2e38", lw=0.8, zorder=0))
        ax.plot(pos[:, 0], pos[:, 1], np.full(len(pos), z0), color="#07080b", lw=3.5,
                alpha=0.55, zorder=1, solid_capstyle="round")

        polys, colors = scene(seq, pos, eye)
        ax.add_collection3d(Poly3DCollection(polys, facecolors=colors, edgecolors=colors,
                                             linewidths=0.25, zorder=2))

        k = fr["key"]
        fit, moves = history[k]
        lattice = decode(moves)
        state = "folding..." if 0 < fr["t"] < 1 else "best fold"
        info.set_text(f"generation  {round(fr['gen']):>4}   {state}\n"
                      f"fitness     {fit:>4}\n"
                      f"H-H contacts{len(contact_pairs(seq, lattice)):>4}\n"
                      f"collisions  {count_collisions(lattice):>4}\n"
                      + (f"{note:<12}{optimum:>4}   gap {fit - optimum:>3}\n" if optimum is not None else "")
                      + "moves " + "\n      ".join(textwrap.wrap(moves, 30)))
        marker.set_data([fr["gen"]], [np.interp(fr["gen"], gens, fits)])
        return []

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".mp4":
        if not shutil.which("ffmpeg"):
            sys.exit("ffmpeg not found: install it or use an output ending in .gif")
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=4000,
                              extra_args=["-pix_fmt", "yuv420p"])
    else:
        writer, dpi = PillowWriter(fps=min(fps, 20)), 60
    print(f"{len(frames)} frames, {len(keyframes(history))} improvements -> {out}")
    FuncAnimation(fig, draw, frames=len(frames)).save(out, writer=writer, dpi=dpi)
    plt.close(fig)


# ---------- command line ----------
def save_run(seq, dim, seed, profile, out, generations=300, target=None, optimum_note=None):
    alphabet = MOVES_3D if dim == 3 else MOVES_2D
    r = run_ga(seq, alphabet, seed=seed, mutation_profile=profile, generations=generations,
               target_energy=target)
    data = {"seq": seq, "dim": dim, "seed": seed, "profile": profile,
            "energy": r["energy"], "optimum": target, "optimum_note": optimum_note,
            "history": [list(h) for h in r["history"]]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(data))
    print(f"{dim}D GA seed {seed}: energy {r['energy']}, {r['generations']} generations -> {out}")


def make_videos(dims, seed, profile, lengths):
    """One run + one movie per sequence and dimension, with the experiment's targets and budgets.
    Runs go to results/runs/, movies to videos/."""
    from run_experiment import SEQS, TARGETS, GENERATIONS
    for dim in dims:
        for seq in SEQS:
            if lengths and len(seq) not in lengths:
                continue
            name = f"ga_{dim}d_n{len(seq)}_{profile}_seed{seed}"
            run_file = ROOT / "results" / "runs" / f"{name}.json"
            # 3D short sequences are proven optimal here (src/exact_3d.c); the rest are best known
            note = "optimum" if dim == 3 and len(seq) <= 25 else "best known"
            save_run(seq, dim, seed, profile, run_file, GENERATIONS[len(seq)], TARGETS[dim][seq], note)
            render(json.loads(run_file.read_text()), str(ROOT / "videos" / f"{name}.mp4"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run", help="run the GA and save its history as JSON")
    a.add_argument("seq")
    a.add_argument("--dim", type=int, choices=(2, 3), default=3)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--profile", default="uniform")
    a.add_argument("--out", default=str(ROOT / "results" / "ga_run.json"))
    a.add_argument("--generations", type=int, default=300)
    a.add_argument("--target", type=int, help="stop early at this energy")
    b = sub.add_parser("render", help="make the movie from a saved run")
    b.add_argument("run")
    b.add_argument("--out", default=str(ROOT / "videos" / "ga_fold3d.mp4"))
    b.add_argument("--fps", type=int, default=30)
    b.add_argument("--move-frames", type=int, default=24, help="frames per fold change")
    v = sub.add_parser("videos", help="run + render every benchmark sequence into videos/")
    v.add_argument("--dims", type=int, nargs="+", default=[3])
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--profile", default="uniform")
    v.add_argument("--lengths", type=int, nargs="+", help="only these sequence lengths")
    args = ap.parse_args()
    if args.cmd == "run":
        save_run(args.seq, args.dim, args.seed, args.profile, args.out, args.generations, args.target,
                 "target" if args.target is not None else None)
    elif args.cmd == "videos":
        make_videos(args.dims, args.seed, args.profile, args.lengths)
    else:
        render(json.loads(Path(args.run).read_text()), args.out, args.fps, args.move_frames)


if __name__ == "__main__":
    main()
