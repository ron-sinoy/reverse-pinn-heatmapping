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
from deepxde.nn.pytorch.nn import NN
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_sensors() -> np.ndarray:
    return np.load(OUTPUT_DIR / "sensor_data.npy")


def grid_coordinates(config: dict, section: str = "inverse_pinn") -> np.ndarray:
    n = int(config[section]["eval_grid_size"])
    length_m = float(config["domain"]["size_mm"]) * 1e-3
    xs = np.linspace(0.0, length_m, n)
    ys = np.linspace(0.0, length_m, n)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1))).astype("float32")


def sensor_coordinates(sensors: np.ndarray, config: dict) -> np.ndarray:
    n = int(config["domain"]["grid_size"])
    length_m = float(config["domain"]["size_mm"]) * 1e-3
    dx = length_m / (n - 1)
    rows = sensors[:, 0]
    cols = sensors[:, 1]
    return np.column_stack((cols * dx, rows * dx)).astype("float32")


class InverseNet(NN):
    def __init__(self, config: dict) -> None:
        super().__init__()
        inv = config["inverse_pinn"]
        length_m = float(config["domain"]["size_mm"]) * 1e-3
        self.power_scale = float(inv["power_scale_w_m2"])
        self.t_net = dde.nn.FNN(
            [2] + [int(inv["t_hidden_units"])] * int(inv["t_hidden_layers"]) + [1],
            str(inv["activation"]),
            "Glorot uniform",
        )
        self.p_net = dde.nn.FNN(
            [2] + [int(inv["p_hidden_units"])] * int(inv["p_hidden_layers"]) + [1],
            str(inv["activation"]),
            "Glorot uniform",
        )
        self.t_net.apply_feature_transform(lambda x: 2.0 * x / length_m - 1.0)
        self.p_net.apply_feature_transform(lambda x: 2.0 * x / length_m - 1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        delta_t = self.t_net(inputs)
        power = torch.nn.functional.softplus(self.p_net(inputs)) * self.power_scale
        return torch.cat((delta_t, power), dim=1)


def build_model(config: dict, sensors: np.ndarray) -> dde.Model:
    inv = config["inverse_pinn"]
    physics = config["physics"]
    length_m = float(config["domain"]["size_mm"]) * 1e-3
    k = float(physics["thermal_conductivity_w_mk"])
    h = float(physics["convection_coefficient_w_m2k"])
    ambient = float(physics["ambient_temperature_c"])
    source_scale = float(inv["power_scale_w_m2"]) / k

    geom = dde.geometry.Rectangle([0.0, 0.0], [length_m, length_m])

    def pde(x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
        d2t_x = dde.grad.hessian(y, x, component=0, i=0, j=0)
        d2t_y = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dp_x = dde.grad.jacobian(y, x, i=1, j=0)
        dp_y = dde.grad.jacobian(y, x, i=1, j=1)
        pde_residual = (d2t_x + d2t_y + y[:, 1:2] / k) / source_scale
        smooth_scale = length_m / float(inv["power_scale_w_m2"])
        return [pde_residual, dp_x * smooth_scale, dp_y * smooth_scale]

    def robin(_x: np.ndarray, y: torch.Tensor) -> torch.Tensor:
        return -(h / k) * y[:, 0:1]

    sensor_xy = sensor_coordinates(sensors, config)
    sensor_delta = (sensors[:, 2:3] - ambient).astype("float32")
    bc = dde.icbc.RobinBC(geom, robin, lambda _x, on_boundary: on_boundary, component=0)
    data_bc = dde.icbc.PointSetBC(sensor_xy, sensor_delta, component=0)
    data = dde.data.PDE(
        geom,
        pde,
        [bc, data_bc],
        num_domain=int(inv["interior_points"]),
        num_boundary=int(inv["boundary_points"]),
        train_distribution="Hammersley",
        num_test=1024,
    )
    model = dde.Model(data, InverseNet(config))
    model.compile(
        "adam",
        lr=float(inv["learning_rate"]),
        loss_weights=[
            float(inv["pde_weight"]),
            float(inv["p_smoothness_weight"]),
            float(inv["p_smoothness_weight"]),
            float(inv["boundary_weight"]),
            float(inv["data_weight"]),
        ],
    )
    return model


def evaluate_net(net: InverseNet, config: dict) -> tuple[np.ndarray, np.ndarray]:
    coords = grid_coordinates(config)
    n = int(config["inverse_pinn"]["eval_grid_size"])
    ambient = float(config["physics"]["ambient_temperature_c"])
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(coords, dtype=torch.float32)).cpu().numpy()
    temperature = pred[:, 0].reshape(n, n) + ambient
    power = pred[:, 1].reshape(n, n)
    return temperature, power


def save_outputs(
    model: dde.Model,
    config: dict,
    loss_train: np.ndarray,
    temperature_pred: np.ndarray,
    power_pred: np.ndarray,
) -> tuple[float, float, float]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    true_power = np.load(OUTPUT_DIR / "power_map.npy")
    true_temp = np.load(OUTPUT_DIR / "ground_truth.npy")
    np.savez(OUTPUT_DIR / "inverse_predictions.npz", temperature=temperature_pred, power=power_pred)
    torch.save(
        {
            "config": config,
            "state_dict": model.net.state_dict(),
            "loss_train": loss_train,
        },
        OUTPUT_DIR / "inverse_pinn_model.pt",
    )

    p_error = np.abs(power_pred - true_power)
    t_error = np.abs(temperature_pred - true_temp)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    plots = [
        (axes[0, 0], true_power, "True P", "inferno"),
        (axes[0, 1], power_pred, "Recovered P", "inferno"),
        (axes[0, 2], p_error, "|P Error|", "viridis"),
        (axes[1, 0], true_temp, "True T", "magma"),
        (axes[1, 1], temperature_pred, "Recovered T", "magma"),
        (axes[1, 2], t_error, "|T Error|", "viridis"),
    ]
    for ax, data, title, cmap in plots:
        im = ax.imshow(data, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "inverse_result.png", dpi=200)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    labels = ["PDE", "P smooth x", "P smooth y", "Boundary", "Sensor data"]
    for idx, label in enumerate(labels):
        if loss_train.size and idx < loss_train.shape[1]:
            ax2.plot(loss_train[:, idx], label=label, linewidth=1.2)
    ax2.set_title("Inverse PINN Loss Curve")
    ax2.set_xlabel("Checkpoint")
    ax2.set_ylabel("Loss")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "inverse_loss_curve.png", dpi=200)
    plt.close(fig2)

    rel_p = np.linalg.norm(power_pred - true_power) / np.linalg.norm(true_power)
    rel_t = np.linalg.norm(temperature_pred - true_temp) / np.linalg.norm(true_temp)
    final_data_loss = float(loss_train[-1, 4]) if loss_train.size and loss_train.shape[1] > 4 else float("nan")
    return float(rel_p), float(rel_t), final_data_loss


def maybe_wandb(config: dict, metrics: dict[str, float]) -> None:
    inv = config["inverse_pinn"]
    if not bool(inv.get("use_wandb", False)):
        return
    try:
        import wandb
    except ImportError:
        return
    run = wandb.init(project="reverse-pinn", mode=str(inv.get("wandb_mode", "offline")), config=config)
    wandb.log(metrics)
    run.finish()


def main() -> None:
    config = load_config()
    sensors = load_sensors()
    seed = int(config["training"]["seed"])
    dde.config.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = build_model(config, sensors)
    losshistory, _ = model.train(
        iterations=int(config["inverse_pinn"]["epochs"]),
        display_every=max(1, int(config["inverse_pinn"]["epochs"]) // 20),
        verbose=1,
    )
    loss_train = np.asarray(losshistory.loss_train, dtype=float)
    temperature_pred, power_pred = evaluate_net(model.net, config)
    rel_p, rel_t, final_data_loss = save_outputs(model, config, loss_train, temperature_pred, power_pred)
    maybe_wandb(config, {"relative_l2_power": rel_p, "relative_l2_temperature": rel_t, "final_sensor_data_loss": final_data_loss})
    print(f"relative_l2_power={rel_p:.6f}")
    print(f"relative_l2_temperature={rel_t:.6f}")
    print(f"final_sensor_data_loss={final_data_loss:.6e}")


if __name__ == "__main__":
    main()
