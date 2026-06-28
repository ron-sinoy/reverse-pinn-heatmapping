# agents.md — Processor Heatmap Inverse PINN

## Project Goal
Build an inverse Physics-Informed Neural Network (PINN) that
reconstructs a full 2D processor die heatmap from a small number of sparse,
synthetic sensor readings, using the steady-state heat equation as a physics
constraint.
## Base Rule
- check work.md
- Keep code minimal and readable 
- Log every completed stage in `agent_log.md` (see logging convention below).

## Stack
- **Python 3.11+** — base language -dont use 3.14
- **PyTorch** — PINN model, training, autodiff backend
- **DeepXDE** — PINN framework (sampling, BC/IC handling, inverse-problem API), backend = PyTorch
- **NumPy / SciPy** — ground truth FDM solve (`scipy.sparse`, `scipy.sparse.linalg.spsolve`)
- **YAML (PyYAML)** — configuration management (domain params, hyperparameters)
- **Weights & Biases (W&B)** — experiment tracking, loss curves, run comparison
- **Matplotlib / Plotly** — final visualization


## Stages — Overall
1. Problem formulation
2. Ground truth generation (NumPy/SciPy FDM)
3. Synthetic sensor sampling
4. Forward PINN sanity check
5. Inverse PINN training
6. Validation against ground truth
7. Visualization
8. Writeup/README

## File/Folder Structure
processor-heatmap-pinn/
├── agents.md
├── agent_log.md
├── work.md
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml             # domain size, block coords, P values, sensor count, noise, hyperparams
├── docs/
│   └── problem_formulation.md
├── src/
│   ├── power_map.py             # Stage 2a: floorplan/P(x,y) generation
│   ├── ground_truth.py          # Stage 2b: FDM solver (uses power_map output)
│   ├── sensors.py               # Stage 3: sampling
│   ├── forward_pinn.py          # Stage 4
│   ├── inverse_pinn.py          # Stage 5
│   ├── validate.py              # Stage 6
│   └── viz.py                   # Stage 7
├── outputs/
│   ├── power_map.npy
│   ├── power_map.png
│   ├── ground_truth.npy
│   ├── ground_truth.png
│   ├── sensor_data.npy
│   ├── forward_pinn_check.png
│   ├── inverse_result.png
│   └── validation_metrics.txt
## Do-Not-Touch / Priorities
- **Mandatory finish: Stages 1–6.
- Ask me on chat if any doubt is there, don't assume or hallucinate.

## Known Unknowns / Assumptions (Fixed Values)
Use these exact values across all stages — do not redefine per-file.

| Parameter                  | Value                                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------ |
| Die domain size            | 20mm × 20mm square                                                                                     |
| Grid resolution            | 100 × 100                                                                                              |
| Thermal conductivity (k)   | 150 W/m·K (silicon), uniform across die                                                                |
| Boundary condition         | Convective (Robin) on all edges, h assumed, T_ambient = 25°C                                           |
| Power map (ground truth P) | Block-structured floorplan: 4 core blocks (high P) + 1 cache block (low-moderate P) + edges/I-O (~0 P) |
| Number of sensors (n)      | 15                                                                                                     |
| Sensor placement           | Random fixed points (seeded) across domain                                                             |
| Sensor noise               | Gaussian, σ = 1.5°C                                                                                    |
| PDE form                   | Steady-state only: ∇²T = -P/k                                                                          |
| PINN unknown               | P(x,y) only (k held fixed/known)                                                                       |
## Power Map (P) Definition — Implementation Instructions

Define P(x,y) as a 100×100 NumPy array representing a simplified chip floorplan.

### Layout rules
- Domain: 100×100 grid representing 20mm × 20mm die.
- Place 4 core blocks + 1 cache block. Core blocks should be small relative
  to the die (roughly 10-15% of grid width/height each side). Cache block
  should be one of the largest contiguous regions (larger than any single
  core block).
- Leave a thin margin (5-10 grid cells) of near-zero P between any block
  and the domain boundary — no block should touch the edge directly.
- Keep gaps between blocks small/tight (a few grid cells), not large empty
  space — real floorplans are densely packed, not sparsely separated.
- Blocks must not overlap.

### Power magnitude rules
- Assign each core block a high, roughly similar P value, EXCEPT vary one
  core's value noticeably higher or lower than the other three (asymmetric,
  not all identical) — simulates uneven core utilization.
- Assign the cache block a lower P value than any core — target roughly
  1/4 to 1/3 of average core P.
- All remaining grid cells (interconnect/I-O/margins) = 0 or near-0 P.

### Implem- Keep gaps between blocks small/tight (a few grid cells), not large emptyentation
- Use array slicing: `P[r1:r2, c1:c2] = value` per block, no smoothing/
  Gaussian blending at edges (flat blocks as defined).
- Store block coordinates and assigned P values in the config file
  (`config.yaml`), not hardcoded inline — so values can be tuned without
  touching code.
- Output: save the resulting P(x,y) array to `outputs/power_map.npy`
  and a heatmap plot to `outputs/power_map.png` for visual sanity check
  before proceeding to the FDM ground-truth solve.

### Validation check before moving to next stage
- Visually confirm in the saved plot: blocks are clearly visible, distinct
  P levels (core > cache > background), no overlaps, no block touching
  domain edge.
## Logging Convention (agent_log.md)
Each completed stage gets one entry: 
-if a edit is made in a single file again, just create a new log, never edit old log 
```
[Stage N - Name] <timestamp>
Done: <1-2 lines, what was built>
Assumptions: <anything decided on the fly, not already in agents.md>
Artifacts: <filenames produced>
Issues: <blockers/workarounds, or "none">
Status: done / partial (+ what's left)
Next-stage note: <what the next stage should load/expect>
```
