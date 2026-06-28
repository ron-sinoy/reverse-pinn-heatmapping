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
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
import matplotlib.pyplot as plt


DEFAULT_CONFIG = {
    "domain": {"size_mm": 20.0, "grid_size": 100},
    "physics": {
        "thermal_conductivity_w_mk": 150.0,
        "ambient_temperature_c": 25.0,
        "convection_coefficient_w_m2k": 200.0,
    },
    "floorplan": {
        "cores": [
            {"name": "core_1", "rows": [12, 23], "cols": [12, 23], "power_w_m2": 16000000.0},
            {"name": "core_2", "rows": [12, 23], "cols": [27, 38], "power_w_m2": 16000000.0},
            {"name": "core_3", "rows": [27, 38], "cols": [12, 23], "power_w_m2": 21000000.0},
            {"name": "core_4", "rows": [27, 38], "cols": [27, 38], "power_w_m2": 16000000.0},
        ],
        "cache": {"name": "cache", "rows": [18, 42], "cols": [48, 74], "power_w_m2": 5500000.0},
    },
    "sensors": {"count": 15, "noise_sigma_c": 1.5, "seed": 42},
    "training": {"seed": 7},
        "forward_pinn": {
            "hidden_layers": 5,
            "hidden_units": 96,
            "activation": "tanh",
            "epochs": 4000,
            "learning_rate": 0.001,
            "interior_points": 2048,
            "boundary_points_per_edge": 256,
            "eval_grid_size": 100,
            "pde_weight": 20.0,
            "boundary_weight": 1.0,
            "data_weight": 1.0,
        },
    }


def _deep_update(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            current = yaml.safe_load(handle) or {}
    else:
        current = {}
    merged = _deep_update(DEFAULT_CONFIG, current)
    if merged != current:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(merged, handle, sort_keys=False)
    return merged


def build_power_map(config: dict) -> np.ndarray:
    grid_size = int(config["domain"]["grid_size"])
    power = np.zeros((grid_size, grid_size), dtype=float)

    floorplan = config["floorplan"]
    for block in floorplan["cores"]:
        r1, r2 = block["rows"]
        c1, c2 = block["cols"]
        power[r1:r2, c1:c2] = float(block["power_w_m2"])

    cache = floorplan["cache"]
    r1, r2 = cache["rows"]
    c1, c2 = cache["cols"]
    power[r1:r2, c1:c2] = float(cache["power_w_m2"])
    return power


def save_outputs(power: np.ndarray) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "power_map.npy", power)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(power, origin="lower", cmap="inferno")
    plt.title("Synthetic Power Map")
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.colorbar(im, label="Power density (W/m^2)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "power_map.png", dpi=200)
    plt.close()


def main() -> None:
    config = load_config()
    power = build_power_map(config)
    save_outputs(power)


if __name__ == "__main__":
    main()
