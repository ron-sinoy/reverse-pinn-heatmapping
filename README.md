# Inverse PINN — Heat Equation Reconstruction
Physics-Informed Neural Network that reconstructs unknown heat source / boundary conditions from sparse sensor data, solving the inverse heat equation.

## Problem

Less sparsely situated sensor datas used to compute with the help of PINNs, no labeled data.

## Approach
- Forward model: 2D/transient heat equation as a soft constraint via PDE residual loss
- Inverse part: unknown parameters treated as trainable variables, optimized jointly with network weights
- Loss = data mismatch (sensor points) + PDE residual + boundary/initial condition residuals (this is what makes PINNs different and more precise)

## Stack

- DeepXDE (PINN framework)
- PyTorch 
- NumPy / Matplotlib for data prep and visualization

**Currently working, trying to optimise results with hyper params tuning**
