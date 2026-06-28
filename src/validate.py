from __future__ import annotations

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
MPLCONFIGDIR = ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
os.environ["DDE_BACKEND"] = "pytorch"

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse_pinn import InverseNet, evaluate_net


def load_prediction_from_model() -> tuple[dict, np.ndarray, np.ndarray]:
    checkpoint = torch.load(OUTPUT_DIR / "inverse_pinn_model.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    net = InverseNet(config)
    net.load_state_dict(checkpoint["state_dict"])
    temperature_pred, power_pred = evaluate_net(net, config)
    np.savez(OUTPUT_DIR / "inverse_predictions.npz", temperature=temperature_pred, power=power_pred)
    return config, temperature_pred, power_pred


def compute_metrics(
    config: dict,
    temperature_pred: np.ndarray,
    power_pred: np.ndarray,
) -> dict[str, float]:
    true_power = np.load(OUTPUT_DIR / "power_map.npy")
    true_temp = np.load(OUTPUT_DIR / "ground_truth.npy")
    sensors = np.load(OUTPUT_DIR / "sensor_data.npy")
    rows = sensors[:, 0].astype(int)
    cols = sensors[:, 1].astype(int)
    sensor_mask = np.zeros(true_temp.shape, dtype=bool)
    sensor_mask[rows, cols] = True
    heldout_mask = ~sensor_mask

    return {
        "relative_l2_power": float(np.linalg.norm(power_pred - true_power) / np.linalg.norm(true_power)),
        "relative_l2_temperature": float(np.linalg.norm(temperature_pred - true_temp) / np.linalg.norm(true_temp)),
        "sensor_rmse_noiseless_temperature": float(
            np.sqrt(np.mean((temperature_pred[rows, cols] - true_temp[rows, cols]) ** 2))
        ),
        "heldout_rmse_temperature": float(np.sqrt(np.mean((temperature_pred[heldout_mask] - true_temp[heldout_mask]) ** 2))),
        "grid_size": float(config["domain"]["grid_size"]),
    }


def save_metrics(metrics: dict[str, float], power_pred: np.ndarray) -> None:
    true_power = np.load(OUTPUT_DIR / "power_map.npy")
    lines = [
        "Validation metrics",
        f"Relative L2 error for P: {metrics['relative_l2_power']:.8f}",
        f"Relative L2 error for T: {metrics['relative_l2_temperature']:.8f}",
        f"RMSE at sensor locations vs noiseless T: {metrics['sensor_rmse_noiseless_temperature']:.8f} C",
        f"RMSE at held-out grid points: {metrics['heldout_rmse_temperature']:.8f} C",
    ]
    (OUTPUT_DIR / "validation_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.abs(power_pred - true_power), origin="lower", cmap="viridis")
    ax.set_title("Absolute Power Recovery Error")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|P_pred - P_true| (W/m^2)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_map.png", dpi=200)
    plt.close(fig)


def main() -> None:
    config, temperature_pred, power_pred = load_prediction_from_model()
    metrics = compute_metrics(config, temperature_pred, power_pred)
    save_metrics(metrics, power_pred)
    for key, value in metrics.items():
        print(f"{key}={value:.8f}")


if __name__ == "__main__":
    main()
