#!/usr/bin/env python3
"""Visualize the memory conflict and its resolution (PCA + anchor_sim).

Loads the per-step (anchor, window) trajectories saved by ``MemoryBank.save_trajectory``
(see ``MEMC_SAVE_TRAJECTORY``), projects the anchor/window vectors into a common 2D PCA
space, and renders a 2x2 figure:

  - top row: 2D PCA of the anchor (orange diamond) vs the drifting recent-window
    (blue, light->dark = early->late). Left = baseline (frozen anchor), right =
    refresh_anchor (anchor re-anchors on conflict).
  - bottom row: anchor_sim over step with the conflict threshold (dashed).

Usage:
  python scripts/plot_memory_conflict.py --baseline /tmp/memc_traj_baseline.npz \
         --refresh /tmp/memc_traj_refresh.npz --out /tmp/memc_conflict.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

# --- reference palette (validated default, light surface) -------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#d03b3b"           # status "critical"
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
THRESHOLD = 0.85


def load_arm(path):
    """Return {eid: {anchor, window, anchor_sim, conflict, step}} from a save_trajectory .npz."""
    data = np.load(path)
    eids = sorted(
        {k[len("eid_"): -len("_anchor")] for k in data.files if k.endswith("_anchor")},
        key=lambda s: int(s) if s.isdigit() else s,
    )
    out = {}
    for eid in eids:
        out[eid] = {
            "anchor": data[f"eid_{eid}_anchor"],        # (T, H)
            "window": data[f"eid_{eid}_window"],        # (T, H)
            "anchor_sim": data[f"eid_{eid}_anchor_sim"],  # (T,)
            "conflict": data[f"eid_{eid}_conflict"],    # (T,) bool
            "step": data[f"eid_{eid}_step"],            # (T,) int
        }
    return out


def fit_pca(arms):
    """Fit 2-component PCA on the union of every anchor and window vector across arms."""
    blocks = []
    for episodes in arms:
        for ep in episodes.values():
            blocks.append(ep["anchor"])
            blocks.append(ep["window"])
    X = np.vstack(blocks)
    mean = X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    var = S ** 2
    ev = var[:2] / var.sum()
    return mean, Vt[:2], ev


def project(X, mean, Vt2):
    return (X - mean) @ Vt2.T


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def unique_anchor_idx(anchor):
    idx = [0]
    for i in range(1, len(anchor)):
        if not np.allclose(anchor[i], anchor[i - 1]):
            idx.append(i)
    return idx


def plot_pca(ax, ep, mean, Vt2, ev, title):
    w = project(ep["window"], mean, Vt2)        # (T, 2)
    a = project(ep["anchor"], mean, Vt2)        # (T, 2)
    step = ep["step"] - ep["step"].min()        # 0-based within episode
    conflict = ep["conflict"]

    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    ax.plot(w[:, 0], w[:, 1], color=INK, lw=0.8, alpha=0.18, zorder=1)
    sc = ax.scatter(
        w[:, 0], w[:, 1], c=step, cmap=cmap, s=10, linewidths=0,
        vmin=step.min(), vmax=step.max(), zorder=2,
    )

    # distinct anchor positions (baseline = 1 fixed point; refresh = re-anchored points)
    a_idx = unique_anchor_idx(ep["anchor"])
    ax.scatter(
        a[a_idx, 0], a[a_idx, 1], marker="D", s=72, color=ORANGE,
        edgecolors=INK, linewidths=0.8, zorder=5, label="anchor",
    )

    # re-anchor jump (only meaningful when the anchor actually moved)
    if len(a_idx) >= 2:
        for j in range(1, len(a_idx)):
            p0 = a[a_idx[j - 1]]
            p1 = a[a_idx[j]]
            ax.add_patch(FancyArrowPatch(
                p0, p1, arrowstyle="-|>", mutation_scale=14,
                color=ORANGE, lw=1.6, zorder=6,
            ))

    # conflict events as red rings around the window position at that step
    if conflict.any():
        ax.scatter(
            w[conflict, 0], w[conflict, 1], facecolors="none", edgecolors=RED,
            s=66, linewidths=1.4, zorder=4, label="conflict",
        )

    # PCA axis labels with explained variance (from the global fit)
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})", color=INK_SEC, fontsize=8)
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})", color=INK_SEC, fontsize=8)
    ax.set_title(title, color=INK, fontsize=10, fontweight="bold")
    style_axis(ax)
    return sc


def plot_sim(ax, ep, title):
    step = ep["step"] - ep["step"].min()
    sim = ep["anchor_sim"]
    conflict = ep["conflict"]

    ax.plot(step, sim, color=BLUE, lw=1.4, zorder=2)
    ax.axhline(THRESHOLD, color=MUTED, ls="--", lw=1.0, zorder=1)
    ax.text(step.min(), THRESHOLD + 0.008, f"threshold {THRESHOLD}",
            color=MUTED, fontsize=7, va="bottom")
    if conflict.any():
        ax.scatter(step[conflict], sim[conflict], facecolors="none",
                   edgecolors=RED, s=40, linewidths=1.1, zorder=3, label="conflict")
    ax.set_xlabel("step (in episode)", color=INK_SEC, fontsize=8)
    ax.set_ylabel("anchor_sim", color=INK_SEC, fontsize=8)
    ax.set_ylim(0.7, 1.02)
    ax.set_title(title, color=INK, fontsize=10, fontweight="bold")
    style_axis(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--refresh", required=True)
    ap.add_argument("--out", default="/tmp/memc_conflict.png")
    args = ap.parse_args()

    arms = [load_arm(args.baseline), load_arm(args.refresh)]
    labels = ["baseline (resolution=none)", "refresh_anchor"]
    mean, Vt2, ev = fit_pca(arms)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.patch.set_facecolor(SURFACE)

    # top row: PCA geometry (use the first episode of each arm; extras still fit PCA)
    for col, (episodes, label) in enumerate(zip(arms, labels)):
        ep = next(iter(episodes.values()))
        sc = plot_pca(axes[0, col], ep, mean, Vt2, ev, label)
        plot_sim(axes[1, col], ep, label)

    fig.colorbar(sc, ax=axes[0, :], location="right", pad=0.01, aspect=40,
                 label="step (early -> late)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
