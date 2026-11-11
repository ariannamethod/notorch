"""Render the C/AML measurement tables; no inference or experiment runs here.

From the repository root: python3 experiments/doe_spores/omni_figure.py
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[2]
data = root / "experiments/doe_spores"

def table(name):
    with (data / name).open() as f:
        return list(csv.DictReader(f, delimiter="\t"))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "svg.hashsalt": "omni-long-experience"})
fig = plt.figure(figsize=(10, 10), facecolor="white", layout="constrained")
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.1, 1.2])
ax = fig.add_subplot(gs[0, :])
colors = {"red": "#b23b3b", "blue": "#266aa5", "text": "#68723c"}
names = {"red": "Red image", "blue": "Blue image", "text": "Text question"}
nights = table("omni-nights.tsv")
for probe in colors:
    rows = [r for r in nights if r["probe"] == probe]
    ax.plot([int(r["turn"]) for r in rows], [float(r["sleep_delta"]) for r in rows],
            marker="o", lw=1.7, color=colors[probe], label=names[probe])
ax.axhline(0, color="#999999", lw=.7)
ax.axvspan(8.5, 16.5, color="#ededed", zorder=0)
ax.set_yscale("symlog", linthresh=1e-5)
ax.set_xlim(2, 26)
ax.set_xticks([4, 8, 12, 16, 20, 24])
ax.set_xlabel("Completed waking turns")
ax.set_ylabel("Post-sleep − pre-sleep\nRed/Blue log-odds margin")
ax.set_title("A  Each night's change, on its current body", loc="left", fontweight="bold")
for x, label in [(5, "Q8"), (12, "F16"), (21, "Q8")]:
    ax.text(x, .96, label, transform=ax.get_xaxis_transform(), ha="center", va="top", color="#444444")
ax.legend(loc="lower left", frameon=False, ncol=3)
ax.grid(axis="y", alpha=.18)

seats = table("omni-seats.tsv")
a = fig.add_subplot(gs[1, 0])
for arm in ["sync"]:
    vitality = np.zeros((6, 24))
    alive = np.zeros((6, 24), dtype=bool)
    for r in seats:
        if not r["state"].startswith(arm + "-"):
            continue
        e, t = int(r["seat"]), int(r["turns"]) - 1
        vitality[e, t] = float(r["vitality"])
        alive[e, t] = float(r["alive"]) != 0
    cm = plt.colormaps["viridis"].copy()
    cm.set_bad("#f0f0f0")
    im = a.imshow(np.ma.masked_where(~alive, vitality), origin="upper", aspect="auto",
                  cmap=cm, vmin=0, vmax=1, extent=(.5, 24.5, 5.5, -.5), interpolation="nearest")
    a.set_title("B  Population with six nights", loc="left", fontweight="bold")
    a.set_yticks(range(6))
    a.set_ylabel("Seat")
    a.set_xlabel("Completed waking turns")
    a.set_xticks([1, 8, 16, 24])
    for x in [8.5, 16.5]:
        a.axvline(x, color="white", lw=1)
fig.colorbar(im, ax=a, shrink=.85, label="Vitality; grey = vacant")

a = fig.add_subplot(gs[1, 1])
distances = table("omni-divergence.tsv")
for seat in range(6):
    rows = [r for r in distances if int(r["seat"]) == seat and float(r["l2_ab"]) > 0]
    a.plot([int(r["turn"]) for r in rows], [float(r["l2_ab"]) for r in rows],
           lw=1.5, marker="." if seat >= 4 else None, label=str(seat))
a.set_yscale("log")
a.set_ylim(1e-6, 5)
a.set_xlim(1, 24)
a.set_xticks([1, 8, 16, 24])
a.set_title("C  Divergence of the two histories", loc="left", fontweight="bold", fontsize=10)
a.set_xlabel("Completed waking turns")
a.set_ylabel("Joint A/B Euclidean distance")
a.legend(title="Seat", ncol=3, frameon=False, fontsize=8, loc="upper left")
a.grid(axis="y", alpha=.18)

ax = fig.add_subplot(gs[2, :])
effects = table("omni-activation.tsv")
xs = np.arange(len(effects))
for offset, key, label, color in [(-.24, "q8_i8_delta", "Q8 / int8 activations", "#74548c"),
                                (0, "q8_float_delta", "Q8 weights / float activations", "#23867c"),
                                (.24, "f16_delta", "F16", "#b57825")]:
    ax.bar(xs + offset, [float(r[key]) for r in effects], width=.22, color=color, label=label)
ax.axhline(0, color="#999999", lw=.7)
ax.set_yscale("symlog", linthresh=1e-4)
ax.set_xticks(xs, [names[r["probe"]] for r in effects])
ax.set_ylabel("Spore − plain\nRed/Blue log-odds margin")
ax.set_title("D  The same episode-8 factors through three arithmetic paths", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="lower left", ncol=3)
ax.grid(axis="y", alpha=.18)
fig.suptitle("A parliament across visual experience, sleep and weights", fontsize=15, fontweight="bold")
svg = root / "research/omni-long-experience.svg"
fig.savefig(svg, metadata={"Date": "2026-11-11"})
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
