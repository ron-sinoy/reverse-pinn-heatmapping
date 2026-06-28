# work.md — Stage 5: Inverse PINN Training

## Objective
Train an inverse PINN that recovers the unknown power map P(x,y) using
only the sparse, noisy sensor data (from Stage 3) plus the PDE
constraint — without ever being given P directly. Validate the result
against the true P(x,y) (known only to us, hidden from the PINN).

## Tasks
1. Load `outputs/sensor_data.npy` (sparse noisy T readings — this is
   the ONLY temperature data the model is allowed to use).
2. Load `outputs/power_map.npy` ONLY for later validation (Stage 6) —
   must NOT be fed into training in any form.
3. Build two networks:
   - Network 1 (T-net): same architecture as Stage 4's forward PINN,
     input (x,y) → output predicted T(x,y).
   - Network 2 (P-net): a second small MLP, input (x,y) → output
     predicted P(x,y). This is the trainable unknown.
4. Define loss terms:
   - PDE residual loss: enforce `∇²T_net - (-P_net/k) = 0` at sampled
     collocation points (same approach as Stage 4, but now P comes from
     P-net instead of being known).
   - Boundary loss: same Robin BC term as Stage 4, applied to T_net.
   - Data loss: match T_net predictions to the 15 sparse sensor
     readings from `sensor_data.npy` (mean squared error).
   - Total loss = PDE loss + boundary loss + data loss (weighted sum;
     data loss may need higher weight since it's the only real
     constraint on absolute temperature scale).
5. Optional regularization: add smoothness or non-negativity penalty on
   P_net output (P should not be negative — heat sources only).
6. Train for a fixed number of epochs (start with same range as Stage 4,
   e.g. 10000–20000; inverse problems often need more epochs to
   converge than forward problems).
7. After training, evaluate both T_net and P_net on the full 100×100
   grid.
8. Save outputs:
   - `outputs/inverse_pinn_model.pt` (trained T-net and P-net weights)
   - `outputs/inverse_result.png` (P_net prediction vs true power_map.npy,
     side-by-side, plus T_net prediction vs ground_truth.npy)
   - `outputs/inverse_loss_curve.png` (training loss over epochs,
     broken down by component: PDE, boundary, data)

## Validation Check 
---

**Validation Check (run once, report only — do not retrain)**

Run training and evaluation **one time**. Report the results against the following checks regardless of outcome — do **not** retrain, adjust hyperparameters, or re-run training to try to improve any of these. Save outputs and log the actual result as-is. Manual review/re-running will be done separately later if needed.

- Visual check: P_net output should show hotspots roughly aligned with the true core/cache block locations, even if blurred/imprecise. **Report what was observed, even if alignment is poor or absent.**
- Loss curve shows convergence across all three loss components, not just one dominating while others stay flat/diverging. **Report the actual shape of each component's curve, even if one or more failed to converge.**
- Sensor data loss specifically should be low (model fits the 15 known points reasonably well) — if not, model hasn't learned to use the data constraint at all. **Report the final data loss value either way.**

## Assumptions Used
- P-net architecture: smaller than T-net is acceptable (e.g. 3–4 hidden
  layers, 32–64 units) since P is expected to be simpler/lower-frequency
  than T in some regions, but blocky/sharp in others — note this is a
  known difficulty (sharp P edges vs smooth T diffusion).
- Loss weights: log final chosen weights in agent_log.md; expect this
  stage may require manual tuning since the problem is underdetermined
  (15 sensors vs 10,000 grid points).
- No access to power_map.npy during training — only for post-hoc
  comparison in Stage 6.

# work.md — Stage 6: Validation Against Ground Truth

## Objective
Quantitatively evaluate how accurately the inverse PINN (Stage 5)
recovered the true power map and temperature field, using standard
IHCP-style error metrics.

## Tasks
1. Load `outputs/inverse_pinn_model.pt` (trained P-net and T-net).
2. Load `outputs/power_map.npy` (true P, used here for the first time
   as a comparison reference) and `outputs/ground_truth.npy` (true T).
3. Evaluate P-net and T-net on the full 100×100 grid.
4. Compute metrics:
   - Relative L2 error for P: `||P_pred - P_true||_2 / ||P_true||_2`
   - Relative L2 error for T: `||T_pred - T_true||_2 / ||T_true||_2`
   - RMSE at sensor locations only (compare T_net prediction at the 15
     sensor coordinates vs the original noiseless ground truth value
     at those points, not the noisy sensor reading — checks whether
     noise was overfit).
   - RMSE at held-out (non-sensor) grid points — broader generalization
     check across the full domain.
5. Save outputs:
   - `outputs/validation_metrics.txt` (plain text, all metrics above
     with labels)
   - Optional: `outputs/error_map.png` (spatial map of |P_pred - P_true|
     across the grid, to visually show where errors concentrate)

## Assumptions Used
- "True" values for comparison are power_map.npy and ground_truth.npy
  exactly as generated in Stage 2 — no modification at this stage.
- Metrics reported as-is, without cherry-picking favorable subsets of
  the grid.

# work.md — Stage 7: Visualization

## Objective
Produce final, presentation-quality visualizations summarizing the
entire pipeline's results, suitable for inclusion in the README/writeup
(Stage 8) and for direct visual inspection.

## Tasks
1. Load all prior saved artifacts: `power_map.npy`, `ground_truth.npy`,
   `sensor_data.npy`, inverse PINN predictions (from Stage 5/6
   evaluation), and `validation_metrics.txt`.
2. Produce a single combined summary figure with subplots:
   - True power map P(x,y)
   - Recovered power map P_pred(x,y) (from inverse PINN)
   - True temperature field T(x,y)
   - Recovered temperature field T_pred(x,y), with sensor locations
     overlaid as markers
   - Absolute error map |P_pred - P_true|
3. Add clear titles, shared colorbars where appropriate, and annotate
   the figure with the headline relative L2 error metric from Stage 6.
4. Optionally produce a clean loss-curve comparison figure (forward PINN
   vs inverse PINN convergence, side-by-side) if useful for the writeup.
5. Save outputs:
   - `outputs/final_summary.png` (the combined multi-panel figure)
   - `outputs/loss_comparison.png` (optional, if generated)

## Validation check
- Figure is legible at normal viewing size (readable titles, labeled
  axes, visible colorbars) — not a cluttered/cramped layout.
- All five panels described above are present and correctly labeled.
- Error metric is visibly annotated, not just buried in a separate file.

## Assumptions Used
- This stage performs no new computation/training — purely visualization
  of existing saved arrays from Stages 2, 3, 5, and 6.
- Single figure preferred over multiple scattered files, for ease of
  use in writeup.