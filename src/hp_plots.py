"""All plots and animations (no algorithm code here). Output goes to ../figures/.

  .venv/bin/python src/hp_plots.py folds [dark]                    fold pictures and GIFs
  .venv/bin/python src/hp_plots.py results results/x.csv [dark]    experiment plots (tables: hp_stats.py)

  fold_2d.png / fold_3d.png   best fold of a 10-mer (brute force)
  fold_2d.gif / fold_3d.gif   same folds built move by move
  ga_2d.gif / ga_3d.gif       GA best fold over the generations (20-mer benchmark)

Experiment plots go to figures/<CSV name>/ (e.g. figures/pilot/, figures/final/); "dark" adds _dark to the file names:
  energy.png  final energy by mutation profile
  evaluations.png, generations.png   cost to reach the target (successful runs only)
  curves.png  median best fitness per generation (band = interquartile range when up to 5 profiles)
  dimension.png  2D vs 3D final energy (uniform profile)

  .venv/bin/python src/hp_plots.py diversity results/diversity.csv [dark]   population diversity plots
  figures/diversity/over_time.png   population diversity per generation (median over runs)
  figures/diversity/by_position.png diversity of the final population at each move position

Colour rules (dataviz skill): categorical slots in fixed order, one colour per entity,
validated with validate_palette.js (adjacent pairs, light and dark). Text uses ink
tokens, never a series colour. Grid and axes are solid hairlines.
"""
import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from hp_core import (decode, contact_pairs, count_collisions, brute_force,
                     MOVES_2D, MOVES_3D)
from hp_ga import run_ga

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
MOVE_NAMES = {"E": "forward", "B": "left", "J": "right", "F": "up", "L": "down"}

# ---------- theme: the values of the reference palette (light and dark steps) ----------
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                          "#008300", "#6250d6", "#e34948"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
                          "#0ca30c", "#9085e9", "#e66767"]),
}
THEME = dict(THEMES["light"], name="light")
# fixed slot per entity (never by rank): the baseline is slot 1
# All 8 slots pass the validator on adjacent pairs, light and dark. Eight series is the token
# ceiling: read the summary table (hp_stats.py) for exact values.
PROFILES = ["uniform", "tail", "tail_exp", "head", "adaptive", "adaptive_rev", "adaptive_exp", "adaptive_exp_rev"]
DIMS = {2: 0, 3: 1}   # 2D = slot 1, 3D = slot 2
H_SLOT, P_SLOT = 1, 0  # fold pictures: H = orange, P = blue


def colour(slot):
    return THEME["series"][slot]


def profile_colour(p):
    return colour(PROFILES.index(p))


def set_theme(name):
    THEME.clear()
    THEME.update(THEMES[name], name=name)
    plt.rcParams.update({
        "figure.facecolor": THEME["surface"], "axes.facecolor": THEME["surface"],
        "savefig.facecolor": THEME["surface"],
        "text.color": THEME["ink"], "axes.labelcolor": THEME["ink2"],
        "axes.edgecolor": THEME["axis"], "axes.linewidth": 1,
        "xtick.color": THEME["muted"], "ytick.color": THEME["muted"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": THEME["grid"],
        "grid.linewidth": 1, "grid.linestyle": "-",
        "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 10, "axes.titlelocation": "left",
        "legend.frameon": False, "lines.solid_capstyle": "round",
    })


def plot_path(prefix, name):
    """figures/<csv name>/<name>.png, so pilot and final plots live in separate folders."""
    folder = FIG_DIR / prefix
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}.png"


def suffix():
    return "_dark" if THEME["name"] == "dark" else ""


def save(fig, path, dpi=130):
    path = Path(path)
    fig.savefig(path.with_name(path.stem + suffix() + path.suffix), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------- fold pictures ----------
def fold_legend(fig):
    """Names for the two residue types and the contact line; identity never rests on colour alone
    because every residue carries its number and the legend names the colours."""
    ring = THEME["surface"]
    handles = [
        Line2D([], [], marker="o", ls="", ms=11, mfc=colour(H_SLOT), mec=ring, mew=2, label="H (hydrophobic)"),
        Line2D([], [], marker="o", ls="", ms=11, mfc=colour(P_SLOT), mec=ring, mew=2, label="P (polar)"),
        Line2D([], [], color=colour(H_SLOT), lw=3, label="H-H contact"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, labelcolor=THEME["ink2"])


def new_axes(dim):
    fig = plt.figure(figsize=(6, 6.2))
    ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)
    fold_legend(fig)
    return fig, ax


def plot_fold(ax, seq, positions, dim, title, limits=None, highlight_last=False):
    """Draw chain, H-H contacts and numbered residues on ax."""
    ax.clear()
    P = np.array(positions)[:, :dim]
    coords = lambda idx: [P[idx, d] for d in range(dim)]
    ax.plot(*coords(slice(None)), "-", color=THEME["muted"], lw=2, zorder=1)
    for i, j in contact_pairs(seq, positions):
        ax.plot(*coords([i, j]), "-", color=colour(H_SLOT), lw=3, zorder=2)
    colors = [colour(H_SLOT) if s == "H" else colour(P_SLOT) for s in seq[:len(P)]]
    extra = {"depthshade": False} if dim == 3 else {}   # no depth fading: colour = residue type only
    ax.scatter(*coords(slice(None)), c=colors, s=260, edgecolors=THEME["surface"],
               linewidths=2, zorder=3, **extra)
    if highlight_last:
        ax.scatter(*coords([-1]), facecolors="none", edgecolors=THEME["ink"], s=620,
                   linewidths=2, zorder=3)
    for i, p in enumerate(P):
        ax.text(*p, str(i + 1), color="#ffffff", ha="center", va="center", fontsize=8, zorder=4)
    lo, hi = limits if limits is not None else (P.min(0) - 0.5, P.max(0) + 0.5)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    axes = [ax.xaxis, ax.yaxis] + ([ax.zaxis] if dim == 3 else [])
    for a in axes:                       # lattice grid, no tick numbers (coordinates carry no meaning)
        a.set_major_locator(MultipleLocator(1)); a.set_ticklabels([])
    if dim == 3:
        ax.set_zlim(lo[2], hi[2]); ax.set_box_aspect(hi - lo)
        for a in axes:
            a.set_pane_color(THEME["surface"])
            a._axinfo["grid"].update(color=THEME["grid"], linewidth=1)
            a._axinfo["axisline"].update(color=THEME["axis"])
        ax.tick_params(length=0)
    else:
        ax.set_aspect("equal"); ax.tick_params(length=0)
    ax.set_title(title, color=THEME["ink"], fontsize=9, loc="left")


def save_fold_png(seq, moves, dim, path):
    positions = decode(moves)
    fig, ax = new_axes(dim)
    c = len(contact_pairs(seq, positions))
    plot_fold(ax, seq, positions, dim,
              f"{dim}D fold of {seq}\nmoves: {' '.join(moves)}\n"
              f"H-H contacts: {c}  =>  energy {-c}")
    save(fig, path)


def save_gif(fig, draw_frame, n_frames, path, fps):
    path = Path(path)
    path = path.with_name(path.stem + suffix() + path.suffix)
    FuncAnimation(fig, draw_frame, frames=n_frames).save(path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)


def animate_build(seq, moves, dim, path, spin=True, hold_frames=12):
    """GIF: the fold is built one move at a time (3D view rotates)."""
    full = np.array(decode(moves))[:, :dim]
    limits = (full.min(0) - 0.5, full.max(0) + 0.5)
    fig, ax = new_axes(dim)

    def draw_frame(f):
        k = min(f, len(moves))  # moves applied so far; k+2 residues placed
        placed = decode(moves[:k])
        c = len(contact_pairs(seq[:k + 2], placed))
        step = "start: residues 1-2 fixed" if k == 0 else \
            f"move {k}: {moves[k-1]} ({MOVE_NAMES[moves[k-1]]}) -> place residue {k + 2}"
        plot_fold(ax, seq[:k + 2], placed, dim, f"{seq}   {dim}D\n{step}\nH-H contacts so far: {c}",
                  limits, highlight_last=True)
        if dim == 3 and spin:
            ax.view_init(25, 30 + 3 * f)

    save_gif(fig, draw_frame, len(moves) + 1 + hold_frames, path, fps=1.5)


def animate_evolution(seq, history, dim, path, hold_frames=6):
    """GIF: GA best fold, one frame each time the best fitness improves."""
    frames = [(g, fit, moves) for g, (fit, moves) in enumerate(history)
              if g == 0 or fit < history[g - 1][0]]
    frames += [frames[-1]] * hold_frames
    fig, ax = new_axes(dim)

    def draw_frame(f):
        g, fit, moves = frames[f]
        positions = decode(moves)
        c, col = len(contact_pairs(seq, positions)), count_collisions(positions)
        plot_fold(ax, seq, positions, dim,
                  f"GA {dim}D, {seq}\ngeneration {g}: fitness {fit}\n"
                  f"H-H contacts {c}, collisions {col}")
        if dim == 3:
            ax.view_init(25, 30 + 8 * f)

    save_gif(fig, draw_frame, len(frames), path, fps=2)


# ---------- experiment data ----------
def load_rows(path):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({"profile": r["profile"], "dim": int(r["dim"]), "seq": r["seq"],
                     "success": r["success"] == "1",
                     "energy": int(r["energy"]) if r["energy"] else None,
                     "evaluations": int(r["evaluations"]), "generations": int(r["generations"]),
                     "curve": [float(x) for x in r["curve"].split(";")]})
    return rows


def profiles_of(rows):
    return [p for p in PROFILES if any(r["profile"] == p for r in rows)]


def quartiles(values):
    return np.percentile(values, [25, 50, 75]) if len(values) else (np.nan,) * 3


# ---------- chart pieces ----------
def style_axis(ax, ylabel, integer=False):
    ax.set_ylabel(ylabel, color=THEME["ink2"])
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    ax.margins(y=0.1)   # keep values at the optimum off the axis edge
    if integer:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def compact(ax):
    """Full numbers below 10,000; compact (10K) on an axis that reaches it - one style per axis."""
    if ax.get_ylim()[1] >= 10000:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}K" if v else "0"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def distribution(ax, x, values, color, width=0.5):
    """Thin box: IQR as a 25% wash, median as a 2px line, whiskers = min..max (1px)."""
    if not len(values):
        return
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    ax.vlines(x, min(values), max(values), color=color, lw=1, zorder=2)
    ax.bar(x, q3 - q1, bottom=q1, width=width, color=color, alpha=0.25, lw=0, zorder=2)
    ax.hlines(med, x - width / 2, x + width / 2, color=color, lw=2, zorder=3)
    ax.scatter([x], [med], s=40, color=color, edgecolors=THEME["surface"], linewidths=2, zorder=4)


def groups_axis(ax, groups, ticklabels=None):
    """groups: {profile: values}. One slot per profile in fixed order; an empty group stays empty."""
    names = list(groups)
    for i, p in enumerate(names):
        distribution(ax, i, groups[p], profile_colour(p))
    ax.set_xticks(range(len(names)), ticklabels or names, fontsize=7, rotation=45, ha="right",
                  rotation_mode="anchor")
    ax.set_xlim(-0.6, len(names) - 0.4)
    if not any(len(v) for v in groups.values()):
        ax.text(0.5, 0.5, "no successful run", ha="center", color=THEME["muted"], transform=ax.transAxes)


def panels(rows, draw, title, path, legend=None, subtitle="whiskers min-max, box = interquartile range, line = median"):
    """One panel per (dimension, sequence); draw(ax, rows_of_that_panel)."""
    seqs = sorted({r["seq"] for r in rows}, key=len)
    fig, axes = plt.subplots(2, len(seqs), figsize=(4.6 * len(seqs), 8), squeeze=False)
    for i, dim in enumerate((2, 3)):
        for j, seq in enumerate(seqs):
            ax = axes[i][j]
            draw(ax, [r for r in rows if r["dim"] == dim and r["seq"] == seq])
            ax.set_title(f"{dim}D, n={len(seq)}", color=THEME["ink"], fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=THEME["ink"])
    fig.text(0.01, 0.945, subtitle, fontsize=8, color=THEME["muted"])
    fig.tight_layout(rect=(0, 0, 1, 0.93 if legend else 0.94))
    if legend:
        fig.legend(handles=legend, loc="upper right", ncol=len(legend), fontsize=8,
                   labelcolor=THEME["ink2"], bbox_to_anchor=(0.99, 0.995))
    save(fig, path, dpi=110)


def profile_legend(rows):
    return [Patch(facecolor=profile_colour(p), label=p) for p in profiles_of(rows)]


# ---------- experiment plots ----------
def plot_energy(rows, prefix):
    def draw(ax, sub):
        groups_axis(ax, {p: [r["energy"] for r in sub if r["profile"] == p and r["energy"] is not None]
                         for p in profiles_of(rows)})
        style_axis(ax, "final energy (lower is better)", integer=True)
    panels(rows, draw, "Final energy by mutation profile", plot_path(prefix, "energy"),
           profile_legend(rows))


def plot_cost(rows, prefix, key):
    def draw(ax, sub):
        names = profiles_of(rows)
        groups = {p: [r[key] for r in sub if r["profile"] == p and r["success"]] for p in names}
        # success counts sit in the tick label, so a box built from few runs is never read alone
        counts = [f"{p}\n{len(groups[p])}/{sum(r['profile'] == p for r in sub)} ok" for p in names]
        groups_axis(ax, groups, counts)
        style_axis(ax, key)
        top = max((max(v) for v in groups.values() if v), default=1)
        ax.set_ylim(0, top * 1.1)   # headroom, so a lone run at the maximum is not clipped
        compact(ax)
        if not any(groups.values()):   # nothing to scale: no misleading ticks
            ax.set_yticks([])
    panels(rows, draw, f"{key.capitalize()} to reach the target",
           plot_path(prefix, key), profile_legend(rows),
           subtitle="successful runs only; n ok = successful runs / all runs; box = interquartile range, line = median")


def plot_curves(rows, prefix):
    def draw(ax, sub):
        for p in profiles_of(rows):
            curves = [r["curve"] for r in sub if r["profile"] == p]
            if not curves:
                continue
            n = max(len(c) for c in curves)
            padded = np.array([c + [c[-1]] * (n - len(c)) for c in curves])  # stopped early = stays at best
            q1, med, q3 = np.percentile(padded, [25, 50, 75], axis=0)
            col = profile_colour(p)
            if len(profiles_of(rows)) <= 5:   # bands of 8 overlapping profiles only muddy the chart
                ax.fill_between(range(n), q1, q3, color=col, alpha=0.10, lw=0)
            ax.plot(med, color=col, lw=2)
            ax.scatter([n - 1], [med[-1]], s=40, color=col, edgecolors=THEME["surface"], linewidths=2, zorder=4)
        ax.set_xlabel("generation")
        style_axis(ax, "best fitness (lower is better)", integer=True)
        ax.grid(axis="x", visible=False)
    panels(rows, draw, "Convergence", plot_path(prefix, "curves"), profile_legend(rows),
           subtitle="line = median over runs (band = interquartile range when up to 5 profiles)")


def plot_dimension(rows, prefix):
    seqs = sorted({r["seq"] for r in rows}, key=len)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for dim, k in DIMS.items():
        for i, s in enumerate(seqs):
            vals = [r["energy"] for r in rows if r["dim"] == dim and r["seq"] == s
                    and r["profile"] == "uniform" and r["energy"] is not None]
            distribution(ax, i * 3 + k, vals, colour(k), width=0.8)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(seqs))], [f"n={len(s)}" for s in seqs])
    style_axis(ax, "final energy (lower is better)", integer=True)
    ax.legend(handles=[Patch(facecolor=colour(k), label=f"{d}D") for d, k in DIMS.items()],
              loc="lower right", fontsize=8, labelcolor=THEME["ink2"])
    ax.set_title("2D vs 3D final energy, uniform mutation", color=THEME["ink"], fontsize=12,
                 fontweight="bold", loc="left", pad=22)
    ax.text(0, 1.03, "whiskers min-max, box = interquartile range, line = median",
            transform=ax.transAxes, fontsize=8, color=THEME["muted"])
    fig.tight_layout()
    save(fig, plot_path(prefix, "dimension"), dpi=110)


def result_plots(path):
    rows, prefix = load_rows(path), Path(path).stem
    plot_energy(rows, prefix)
    plot_cost(rows, prefix, "evaluations")
    plot_cost(rows, prefix, "generations")
    plot_curves(rows, prefix)
    plot_dimension(rows, prefix)
    print("plots written to", FIG_DIR / prefix, "(numbers: run src/hp_stats.py)")


def plot_diversity(path):
    from hp_diversity import load, STEP
    rows, prefix = load(path), Path(path).stem

    def over_time(ax, sub):
        for p in profiles_of(rows):
            curves = np.array([r["hamming"] for r in sub if r["profile"] == p])
            if len(curves):
                med = np.median(curves, axis=0)
                ax.plot(np.arange(len(med)) * STEP, med, color=profile_colour(p), lw=2)
        ax.set_xlabel("generation")
        style_axis(ax, "population diversity")
        ax.set_ylim(bottom=0)

    def by_position(ax, sub):
        for p in profiles_of(rows):
            pos = np.array([r["positions"] for r in sub if r["profile"] == p])
            if len(pos):
                ax.plot(np.arange(1, pos.shape[1] + 1), np.median(pos, axis=0), color=profile_colour(p), lw=2)
        ax.set_xlabel("move position (1 = first move, last = end of the chain)")
        style_axis(ax, "diversity at the end of the run")
        ax.set_ylim(bottom=0)

    note = "pairwise Hamming distance of the move strings / length, median over runs"
    panels(rows, over_time, "Population diversity during the run", plot_path(prefix, "over_time"),
           profile_legend(rows), subtitle=note)
    panels(rows, by_position, "Diversity of the final population by move position",
           plot_path(prefix, "by_position"), profile_legend(rows), subtitle=note)
    print("plots written to", FIG_DIR / prefix)


def fold_figures():
    small_seq = "HHPHHPHHHH"  # short enough to brute force in 3D (5^8 folds)
    for dim, alphabet in [(2, MOVES_2D), (3, MOVES_3D)]:
        e, moves = brute_force(small_seq, alphabet)
        print(f"{dim}D brute force: energy {e}, moves {moves}")
        save_fold_png(small_seq, moves, dim, FIG_DIR / f"fold_{dim}d.png")
        animate_build(small_seq, moves, dim, FIG_DIR / f"fold_{dim}d.gif")

    ga_seq = "HPHPPHHPHPPHPHHPPHPH"
    for dim, alphabet in [(2, MOVES_2D), (3, MOVES_3D)]:
        r = run_ga(ga_seq, alphabet, seed=0)
        print(f"{dim}D GA: energy {r['energy']}, moves {r['moves']}")
        animate_evolution(ga_seq, r["history"], dim, FIG_DIR / f"ga_{dim}d.gif")


if __name__ == "__main__":
    args = sys.argv[1:]
    set_theme("dark" if args and args[-1] == "dark" else "light")
    args = [a for a in args if a != "dark"]
    if args == ["folds"]:
        fold_figures()
    elif len(args) == 2 and args[0] == "results":
        result_plots(args[1])
    elif len(args) == 2 and args[0] == "diversity":
        plot_diversity(args[1])
    else:
        print(__doc__)
