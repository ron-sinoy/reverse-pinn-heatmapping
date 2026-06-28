from __future__ import annotations

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
OUTPUT_DIR = ROOT / "outputs"
MPLCONFIGDIR = ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_arrays() -> tuple[np.ndarray, np.ndarray]:
    return np.load(OUTPUT_DIR / "power_map.npy"), np.load(OUTPUT_DIR / "ground_truth.npy")


def make_grid(config: dict) -> np.ndarray:
    n = int(config["forward_pinn"]["eval_grid_size"])
    length_m = float(config["domain"]["size_mm"]) * 1e-3
    xs = np.linspace(0.0, length_m, n)
    ys = np.linspace(0.0, length_m, n)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1))).astype("float32")


def power_at_points(x: torch.Tensor, power: torch.Tensor, length_m: float) -> torch.Tensor:
    n = power.shape[0]
    dx = length_m / (n - 1)
    rows = torch.clamp(torch.round(x[:, 1:2] / dx).long(), 0, n - 1)
    cols = torch.clamp(torch.round(x[:, 0:1] / dx).long(), 0, n - 1)
    return power[rows, cols].reshape(-1, 1)


def build_model(config: dict, power: np.ndarray, ground_truth: np.ndarray) -> dde.Model:
    pinn = config["forward_pinn"]
    physics = config["physics"]
    length_m = float(config["domain"]["size_mm"]) * 1e-3
    k = float(physics["thermal_conductivity_w_mk"])
    h = float(physics["convection_coefficient_w_m2k"])
    ambient = float(physics["ambient_temperature_c"])
    source_scale = float(np.max(power) / k)
    power_tensor = torch.tensor(power, dtype=torch.float32)

    geom = dde.geometry.Rectangle([0.0, 0.0], [length_m, length_m])

    def pde(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d2t_x = dde.grad.hessian(y, x, component=0, i=0, j=0)
        d2t_y = dde.grad.hessian(y, x, component=0, i=1, j=1)
        source = power_at_points(x, power_tensor.to(x.device), length_m)
        return (d2t_x + d2t_y + source / k) / source_scale

    def robin(_x: np.ndarray, y: torch.Tensor) -> torch.Tensor:
        return -(h / k) * y[:, 0:1]

    bc = dde.icbc.RobinBC(geom, robin, lambda _x, on_boundary: on_boundary, component=0)
    coords = make_grid(config)
    data_values = (ground_truth.reshape(-1, 1) - ambient).astype("float32")
    data_bc = dde.icbc.PointSetBC(coords, data_values, component=0)
    data = dde.data.PDE(
        geom,
        pde,
        [bc, data_bc],
        num_domain=int(pinn["interior_points"]),
        num_boundary=int(pinn["boundary_points_per_edge"]) * 4,
        train_distribution="Hammersley",
        num_test=1024,
    )
    net = dde.nn.FNN(
        [2] + [int(pinn["hidden_units"])] * int(pinn["hidden_layers"]) + [1],
        str(pinn["activation"]),
        "Glorot uniform",
    )
    net.apply_feature_transform(lambda x: 2.0 * x / length_m - 1.0)
    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=float(pinn["learning_rate"]),
        loss_weights=[
            float(pinn["pde_weight"]),
            float(pinn["boundary_weight"]),
            float(pinn["data_weight"]),
        ],
    )
    return model


def evaluate_model(model: dde.Model, config: dict) -> np.ndarray:
    coords = make_grid(config)
    pred_delta = model.predict(coords).reshape(int(config["forward_pinn"]["eval_grid_size"]), -1)
    return pred_delta + float(config["physics"]["ambient_temperature_c"])


def save_outputs(
    model: dde.Model,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    loss_train: np.ndarray,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.net.state_dict(), OUTPUT_DIR / "forward_pinn_model.pt")

    error = np.abs(prediction - ground_truth)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, data, title, cmap in [
        (axes[0], ground_truth, "FDM Ground Truth", "magma"),
        (axes[1], prediction, "DeepXDE Forward PINN", "magma"),
        (axes[2], error, "Absolute Error", "viridis"),
    ]:
        im = ax.imshow(data, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "forward_pinn_check.png", dpi=200)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    if loss_train.size:
        ax2.plot(loss_train[:, 0], label="PDE", linewidth=1.2)
        ax2.plot(loss_train[:, 1], label="Boundary", linewidth=1.2)
        ax2.plot(loss_train[:, 2], label="Data", linewidth=1.2)
    ax2.set_title("Forward PINN Loss Curve")
    ax2.set_xlabel("Checkpoint")
    ax2.set_ylabel("Loss")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "forward_pinn_loss_curve.png", dpi=200)
    plt.close(fig2)


def main() -> None:
    config = load_config()
    power, ground_truth = load_arrays()
    seed = int(config["training"]["seed"])
    dde.config.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = build_model(config, power, ground_truth)
    losshistory, _ = model.train(
        iterations=int(config["forward_pinn"]["epochs"]),
        display_every=max(1, int(config["forward_pinn"]["epochs"]) // 20),
        verbose=1,
    )
    prediction = evaluate_model(model, config)
    loss_train = np.asarray(losshistory.loss_train, dtype=float)
    save_outputs(model, prediction, ground_truth, loss_train)
    rel_l2 = np.linalg.norm(prediction - ground_truth) / np.linalg.norm(ground_truth)
    print(f"relative_l2_error={rel_l2:.6f}")


if __name__ == "__main__":
    main()
