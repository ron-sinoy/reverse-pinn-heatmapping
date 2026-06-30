# Inverse PINN — Heat Equation Reconstruction

Physics-Informed Neural Network that reconstructs unknown heat source / boundary conditions from sparse sensor data, solving the inverse heat equation.

## Problem

Given sparse, noisy temperature sensor readings, recover the underlying heat equation parameters (source term, BCs, or material coefficients) without solving the full forward problem from scratch — the PDE residual constrains the solution alongside the data.

## Approach

- Forward model: 2D/transient heat equation as a soft constraint via PDE residual loss
- Inverse part: unknown parameters treated as trainable variables, optimized jointly with network weights
- Loss = data mismatch (sensor points) + PDE residual + boundary/initial condition residuals

## Stack

- DeepXDE (PINN framework)
- PyTorch backend
- NumPy / Matplotlib for data prep and visualization
