# Problem Formulation

This project reconstructs a full 2D processor die heatmap from sparse synthetic sensor readings using an inverse physics-informed neural network (PINN).

## Objective

Estimate the spatial power density field `P(x, y)` from observed steady-state temperature samples `T(x, y)` while keeping thermal conductivity fixed and known.

## Fixed Problem Setup

- Die domain: `20 mm x 20 mm`
- Grid resolution: `100 x 100`
- Thermal conductivity: `k = 150 W/m·K`
- Boundary condition: convective (Robin) on all edges
- Ambient temperature: `25°C`
- PDE: steady-state heat equation

## Governing Equation

The thermal field is modeled by:

`∇²T = -P / k`

where:

- `T(x, y)` is the die temperature field
- `P(x, y)` is the unknown power map to be reconstructed
- `k` is uniform and known

## Synthetic Data Assumptions

- Number of sensors: `15`
- Sensor placement: random fixed points with a seeded generator
- Sensor noise: Gaussian, `σ = 1.5°C`
- Ground truth comes from a simplified block-structured floorplan

## Stage-1 Scope

This stage only defines the problem and project scaffold. Later stages will generate the power map, solve the ground-truth temperature field, sample sensor data, and train the inverse PINN.
