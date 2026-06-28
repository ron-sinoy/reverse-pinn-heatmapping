from __future__ import annotations

from pathlib import Path
import os
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
MPLCONFIGDIR = ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metric_label() -> str:
    metrics_path = OUTPUT_DIR / "validation_metrics.txt"
    if not metrics_path.exists():
        return "Relative L2 P: unavailable"
    text = metrics_path.read_text(encoding="utf-8")
    match = re.search(r"Relative L2 error for P:\s*([0-9.eE+-]+)", text)
    if not match:
        return "Relative L2 P: unavailable"
    return f"Relative L2 P: {float(match.group(1)):.4f}"


def save_final_summary() -> None:
    true_power = np.load(OUTPUT_DIR / "power_map.npy")
    true_temp = np.load(OUTPUT_DIR / "ground_truth.npy")
    sensors = np.load(OUTPUT_DIR / "sensor_data.npy")
    pred = np.load(OUTPUT_DIR / "inverse_predictions.npz")
    power_pred = pred["power"]
    temp_pred = pred["temperature"]
    p_error = np.abs(power_pred - true_power)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    panels = [
        (axes[0, 0], true_power, "True Power Map", "inferno"),
        (axes[0, 1], power_pred, "Recovered Power Map", "inferno"),
        (axes[0, 2], p_error, "Absolute Power Error", "viridis"),
        (axes[1, 0], true_temp, "True Temperature Field", "magma"),
        (axes[1, 1], temp_pred, "Recovered Temperature + Sensors", "magma"),
    ]
    for ax, data, title, cmap in panels:
        im = ax.imshow(data, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[1, 1].scatter(
        sensors[:, 1],
        sensors[:, 0],
        c="cyan",
        s=42,
        edgecolors="black",
        linewidths=0.6,
        label="Sensors",
    )
    axes[1, 1].legend(loc="upper right")
    axes[1, 2].axis("off")
    axes[1, 2].text(0.04, 0.70, load_metric_label(), fontsize=16, weight="bold", transform=axes[1, 2].transAxes)
    axes[1, 2].text(
        0.04,
        0.55,
        "Inverse PINN validation\nagainst Stage 2 ground truth",
        fontsize=12,
        transform=axes[1, 2].transAxes,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "final_summary.png", dpi=200)
    plt.close(fig)


def save_loss_comparison() -> None:
    forward = OUTPUT_DIR / "forward_pinn_loss_curve.png"
    inverse = OUTPUT_DIR / "inverse_loss_curve.png"
    if not forward.exists() or not inverse.exists():
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, path, title in [
        (axes[0], forward, "Forward PINN Loss"),
        (axes[1], inverse, "Inverse PINN Loss"),
    ]:
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loss_comparison.png", dpi=200)
    plt.close(fig)


def main() -> None:
    save_final_summary()
    save_loss_comparison()
    print("saved final_summary.png")


if __name__ == "__main__":
    main()
