#!/usr/bin/env python
"""Generate the report's PDF figures (matplotlib, vector).

  fig_trajectory_main.pdf / fig_trajectory_original.pdf
      Circular expanding lineage: ring = round, node = trajectory, curved
      edge = parent->child, color = phase, filled = admitted, halo = beat-parent.
  fig_learning.pdf
      Solo per-round median (no anti-learn) + cumulative-zoo composite rank IC
      (system-level learning) for main vs original.

Yearly-bar figures (fig_yearly_*) are added once report_yearly.json exists.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)

PHASE_COLOR = {"original": "#4C72B0", "mutation": "#55A868", "crossover": "#C44E52"}
ROUND_GAP_OK = {0, 1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}  # main rounds present


def _node_xy(rounds_present, r, angle):
    """radius grows with round INDEX (position in sorted present rounds), not raw
    round number, so the main run's gap (3-7) doesn't leave empty rings."""
    idx = rounds_present.index(r) + 1
    return idx, angle


def trajectory_figure(lineage_path: Path, out: Path, title: str, subtitle: str = ""):
    """Circular expanding lineage: ring = evolution round (radius grows with the
    round's ORDINAL so a resume gap leaves no empty ring), node = trajectory,
    curved chord = parent -> child, colour = phase, filled = admitted, green halo
    = the child beat its best parent."""
    d = json.loads(lineage_path.read_text())
    nodes = d["nodes"]; edges = d["edges"]
    rounds = sorted({n["round"] for n in nodes.values() if n["round"] is not None})
    ring = {r: i + 1 for i, r in enumerate(rounds)}

    # angle: nodes of a round are laid out in phase blocks so the phase mix of
    # each generation is legible as an arc, with a small offset per ring so
    # successive rings do not line up into spokes.
    order = {"original": 0, "mutation": 1, "crossover": 2}
    pos = {}
    for r in rounds:
        ns = [n for n in nodes.values() if n["round"] == r]
        # children inherit their parent's angle so a lineage reads as a radial
        # branch; roots (and unresolvable parents) fall back to phase order.
        def anchor(n):
            pa = [pos[p][1] for p in n["parents"] if p in pos]
            if pa:
                x = np.mean(np.cos(pa)); y = np.mean(np.sin(pa))
                return float(np.arctan2(y, x)) % (2 * np.pi)
            return None
        anchored = [(n, anchor(n)) for n in ns]
        anchored.sort(key=lambda z: (z[1] is None,
                                     z[1] if z[1] is not None else 0.0,
                                     order.get(z[0]["phase"], 9)))
        k = len(anchored)
        for i, (n, a) in enumerate(anchored):
            even = 2 * np.pi * i / max(1, k)
            # blend the parent's angle with an even spread so nodes never overlap
            theta = even if a is None else float(np.angle(
                0.62 * np.exp(1j * a) + 0.38 * np.exp(1j * even)))
            pos[n["id"]] = (ring[r], theta % (2 * np.pi))

    fig = plt.figure(figsize=(7.4, 7.8))
    ax = fig.add_subplot(111, projection="polar")

    # --- edges: quadratic-ish chords in polar space (interpolate r and theta) ---
    segs = []
    for e in edges:
        p_, c_ = e["parent"], e["child"]
        if p_ not in pos or c_ not in pos:
            continue
        r0, t0 = pos[p_]; r1, t1 = pos[c_]
        dt = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi      # shortest way round
        u = np.linspace(0, 1, 18)
        ts = t0 + dt * (u ** 1.35)      # travel outward first, then swing across
        rs = np.linspace(r0, r1, 18)
        segs.append(np.column_stack([ts, rs]))
    if segs:
        ax.add_collection(LineCollection(segs, colors="#9aa5b1", linewidths=0.45,
                                         alpha=0.5, zorder=1))

    # --- nodes ---
    for n in nodes.values():
        if n["id"] not in pos:
            continue
        r, t = pos[n["id"]]
        col = PHASE_COLOR.get(n["phase"], "#999999")
        if n["admitted"]:
            ax.scatter([t], [r], s=46, c=col, edgecolors="white", linewidths=0.7,
                       zorder=4)
        else:
            ax.scatter([t], [r], s=15, c=col, alpha=0.40, linewidths=0, zorder=3)
        if n.get("beat_parent"):
            ax.scatter([t], [r], s=132, facecolors="none", edgecolors="#2f9e44",
                       linewidths=1.1, alpha=0.85, zorder=2)

    # --- rings + round labels ---
    for r in rounds:
        ax.plot(np.linspace(0, 2 * np.pi, 200), [ring[r]] * 200,
                color="#e3e6ea", lw=0.5, zorder=0)
    step = max(1, len(rounds) // 6)
    for r in rounds[::step] + [rounds[-1]]:
        ax.text(0.0, ring[r] + 0.16, f"r{r}", ha="center", va="center", fontsize=6,
                color="#6b7280",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9),
                zorder=6)

    ax.set_ylim(0, len(rounds) + 1.0)
    ax.set_yticklabels([]); ax.set_xticklabels([])
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.grid(False); ax.spines["polar"].set_visible(False)
    ax.set_title(title, pad=30, fontsize=12, fontweight="bold")
    if subtitle:
        ax.text(0.5, 1.055, subtitle, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.5, color="#4b5563")

    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=PHASE_COLOR[p],
                  markersize=8, label=p) for p in ("original", "mutation", "crossover")]
    leg += [Line2D([0], [0], marker="o", color="w", markerfacecolor="#6b7280",
                   markeredgecolor="white", markersize=9, label="admitted"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#6b7280",
                   alpha=0.4, markersize=5, label="rejected"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor="#2f9e44", markersize=11, markeredgewidth=1.2,
                   label="beat parent")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, fontsize=7.5, frameon=False, handletextpad=0.4,
              columnspacing=1.4)
    n_adm = sum(1 for n in nodes.values() if n["admitted"])
    n_beat = sum(1 for n in nodes.values() if n.get("beat_parent"))
    fig.text(0.5, 0.055, f"{d['n_nodes']} trajectories   |   {len(edges)} lineage edges"
                         f"   |   {n_adm} admitted   |   {n_beat} beat parent",
             ha="center", fontsize=7.5, color="#6b7280")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


def learning_figure(learn_path: Path, out: Path):
    d = json.loads(learn_path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # (a) solo per-round median (no anti-learn)
    ax = axes[0]
    for key, col, label in [("main", "#4C72B0", "main"), ("original", "#C44E52", "original")]:
        c = d[key]
        x = c["rounds"]; y = c["median"]
        ax.plot(x, y, "-o", color=col, lw=1.6, ms=4, label=label)
        ax.axhline(np.mean(y), color=col, ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("evolution round"); ax.set_ylabel("per-round median |quality|")
    ax.set_title("(a) No anti-learning: per-round median factor quality", fontsize=10)
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)
    # (b) system-level: cumulative-zoo composite rank_ic (main ledger) vs original best-bar
    ax = axes[1]
    s = d["main"]["system_level"]
    ax.plot(s["zoo_size"], s["rank_ic"], "-o", color="#4C72B0", lw=1.8, ms=3,
            label="main: cumulative-zoo composite rank IC (gate)")
    ax.axhline(0, color="#888888", lw=0.8)
    # original: solo cumulative best bar (its system-level proxy, no gate)
    o = d["original"]
    ax.plot(o["rounds"], o["zoo_best"], "--s", color="#C44E52", lw=1.4, ms=4,
            label="original: cumulative best |Rank IC| (no gate)")
    ax.set_xlabel("main: zoo size (admitted)   |   original: round")
    ax.set_ylabel("rank IC")
    ax.set_title("(b) System-level: zoo composite rises (main learns)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.grid(alpha=0.25)
    fig.suptitle("Learning: main's gate builds a progressively better book; "
                 "original's frontier creeps up at a lower level",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    trajectory_figure(ROOT / "data/results/report_lineage_main.json",
                      FIG / "fig_trajectory_main.pdf",
                      "MAIN (learning-aware)",
                      "every parent link resolves; the gate selects a surviving book")
    trajectory_figure(ROOT / "data/results/report_lineage_original.json",
                      FIG / "fig_trajectory_original.pdf",
                      "ORIGINAL (paper baseline)",
                      "60 trajectories, all 37 parent links resolve")
    learning_figure(ROOT / "data/results/report_learning.json",
                    FIG / "fig_learning.pdf")