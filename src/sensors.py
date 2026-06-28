from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
OUTPUT_DIR = ROOT / "outputs"
MPLCONFIGDIR = ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_ground_truth() -> np.ndarray:
    return np.load(OUTPUT_DIR / "ground_truth.npy")


def sample_sensors(temperature: np.ndarray, config: dict) -> np.ndarray:
    sensor_cfg = config["sensors"]
    count = int(sensor_cfg["count"])
    seed = int(sensor_cfg["seed"])
    sigma = float(sensor_cfg["noise_sigma_c"])

    rng = np.random.default_rng(seed)
    grid_size = temperature.shape[0]
    indices = rng.choice(grid_size * grid_size, size=count, replace=False)
    rows = indices // grid_size
    cols = indices % grid_size
    noisy_values = temperature[rows, cols] + rng.normal(0.0, sigma, size=count)
    return np.column_stack((rows, cols, noisy_values))


def save_outputs(temperature: np.ndarray, sensors: np.ndarray) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "sensor_data.npy", sensors)

    fig, ax = plt.subplots(figsize=(6, 5))
    heat = ax.imshow(temperature, origin="lower", cmap="magma")
    scatter = ax.scatter(
        sensors[:, 1],
        sensors[:, 0],
        c=sensors[:, 2],
        cmap="coolwarm",
        s=65,
        edgecolors="white",
        linewidths=0.8,
    )
    ax.set_title("Sensor Overlay on Ground Truth")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04, label="Ground truth T (°C)")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.10, label="Sensor T (°C)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sensor_overlay.png", dpi=200)
    plt.close(fig)


def main() -> None:
    config = load_config()
    temperature = load_ground_truth()
    sensors = sample_sensors(temperature, config)
    save_outputs(temperature, sensors)


if __name__ == "__main__":
    main()
