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
