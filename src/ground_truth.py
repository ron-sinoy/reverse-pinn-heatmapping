from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
OUTPUT_DIR = ROOT / "outputs"
MPLCONFIGDIR = ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
import matplotlib.pyplot as plt


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_power_map() -> np.ndarray:
    return np.load(OUTPUT_DIR / "power_map.npy")


def build_system(power: np.ndarray, config: dict) -> tuple[sp.csr_matrix, np.ndarray]:
    n = int(config["domain"]["grid_size"])
    size_mm = float(config["domain"]["size_mm"])
    k = float(config["physics"]["thermal_conductivity_w_mk"])
    t_ambient = float(config["physics"]["ambient_temperature_c"])
    h = float(config["physics"]["convection_coefficient_w_m2k"])

    dx = (size_mm * 1e-3) / (n - 1)
    dy = dx
    alpha_x = 1.0 / dx**2
    alpha_y = 1.0 / dy**2
    beta_x = 2.0 * h / (k * dx)
    beta_y = 2.0 * h / (k * dy)

    def idx(i: int, j: int) -> int:
        return i * n + j

    rows = []
    cols = []
    data = []
    rhs = np.zeros(n * n, dtype=float)

    for i in range(n):
        for j in range(n):
            row = idx(i, j)
            diag = 0.0
            rhs[row] = -power[i, j] / k

            if i == 0:
                diag -= 2.0 * alpha_x + beta_x
                rows.append(row)
                cols.append(idx(i + 1, j))
                data.append(2.0 * alpha_x)
                rhs[row] -= beta_x * t_ambient
            elif i == n - 1:
                diag -= 2.0 * alpha_x + beta_x
                rows.append(row)
                cols.append(idx(i - 1, j))
                data.append(2.0 * alpha_x)
                rhs[row] -= beta_x * t_ambient
            else:
                diag -= 2.0 * alpha_x
                rows.append(row)
                cols.append(idx(i - 1, j))
                data.append(alpha_x)
                rows.append(row)
                cols.append(idx(i + 1, j))
                data.append(alpha_x)

            if j == 0:
                diag -= 2.0 * alpha_y + beta_y
                rows.append(row)
                cols.append(idx(i, j + 1))
                data.append(2.0 * alpha_y)
                rhs[row] -= beta_y * t_ambient
            elif j == n - 1:
                diag -= 2.0 * alpha_y + beta_y
                rows.append(row)
                cols.append(idx(i, j - 1))
                data.append(2.0 * alpha_y)
                rhs[row] -= beta_y * t_ambient
            else:
                diag -= 2.0 * alpha_y
                rows.append(row)
                cols.append(idx(i, j - 1))
                data.append(alpha_y)
                rows.append(row)
                cols.append(idx(i, j + 1))
                data.append(alpha_y)

            rows.append(row)
            cols.append(row)
            data.append(diag)

    matrix = sp.csr_matrix((data, (rows, cols)), shape=(n * n, n * n))
    return matrix, rhs


def solve_temperature(power: np.ndarray, config: dict) -> np.ndarray:
    matrix, rhs = build_system(power, config)
    temperature = spla.spsolve(matrix, rhs)
    return temperature.reshape(power.shape)


def save_outputs(temperature: np.ndarray) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "ground_truth.npy", temperature)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(temperature, origin="lower", cmap="magma")
    plt.title("Ground Truth Temperature")
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.colorbar(im, label="Temperature (°C)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ground_truth.png", dpi=200)
    plt.close()


def main() -> None:
    config = load_config()
    power = load_power_map()
    temperature = solve_temperature(power, config)
    save_outputs(temperature)


if __name__ == "__main__":
    main()
