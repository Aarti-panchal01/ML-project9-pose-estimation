"""All Person 1 figures.

    python src/plots.py
Saves to results/figures/:
    fig4_error_true_vs_classified.png   paper Fig 4 equivalent
    fig5_error_label_vs_classified.png  paper Fig 5 equivalent
    training_curve.png
    camera_views.png                    what the follower camera 'sees' at a few poses
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
from aircraft import EDGES
from simulator import project

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "results", "figures")
LABELS = ["x [m]", "y [m]", "z [m]", "phi [deg]"]


def error_histograms(delta, title, fname, bins):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for i, ax in enumerate(axes.flat):
        ax.hist(delta[:, i], bins=bins, color="#2b7bba", edgecolor="#1b4f7a", linewidth=0.3)
        ax.set_xlabel(LABELS[i])
        ax.set_ylabel("count")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)


def training_curve():
    log = np.genfromtxt(os.path.join(ROOT, "results", "training_log.csv"), delimiter=",", names=True)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(log["epoch"], log["train_loss"], color="#c0392b", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="#c0392b")
    ax2 = ax1.twinx()
    ax2.plot(log["epoch"], log["val_accuracy"], color="#2b7bba", label="val accuracy")
    ax2.axhline(0.54, ls="--", color="grey", lw=1)
    ax2.text(log["epoch"][0], 0.545, "paper: 54%", color="grey", fontsize=8)
    ax2.set_ylabel("val accuracy", color="#2b7bba")
    fig.suptitle("ANN training")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "training_curve.png"), dpi=150)
    plt.close(fig)


def draw_camera_view(ax, pose, title=""):
    """Plot the 14 projected features (visible = red, occluded = hollow) + wireframe."""
    _, uv, vis = project(pose, noise=False, return_pixels=True)
    for a, b in EDGES:
        ax.plot(uv[[a, b], 0], uv[[a, b], 1], color="#333", lw=0.8, alpha=0.6)
    ax.scatter(uv[vis, 0], uv[vis, 1], s=22, color="#d62728", zorder=3, label="visible")
    ax.scatter(uv[~vis, 0], uv[~vis, 1], s=22, facecolors="none", edgecolors="#999",
               zorder=3, label="occluded")
    ax.set_xlim(0, C.IMG_W)
    ax.set_ylim(C.IMG_H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def camera_views():
    poses = [
        [-30, 0, 0, 0, 0, 0],
        [-25, 10, -5, 20, 0, 0],
        [-60, -30, 10, -30, 3, -3],
        [-20, 15, 5, 40, 0, 5],
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, p in zip(axes.flat, poses):
        draw_camera_view(ax, p, f"x={p[0]} y={p[1]} z={p[2]} m, roll={p[3]} deg")
    axes.flat[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Follower camera view of the leader's 14 features")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "camera_views.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    camera_views()
    err_path = os.path.join(ROOT, "results", "error_distribution.npz")
    if os.path.exists(err_path):
        e = np.load(err_path)
        error_histograms(e["delta_true"], "Error: true pose vs classified pose (val set)",
                         "fig4_error_true_vs_classified.png", bins=60)
        error_histograms(e["delta_label"], "Error: correct label vs classified label (val set)",
                         "fig5_error_label_vs_classified.png", bins=80)
        training_curve()
    print(f"Figures saved to {os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
