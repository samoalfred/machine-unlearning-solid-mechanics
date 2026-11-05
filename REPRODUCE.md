# Reproducing the results

This guide maps every table and figure in the manuscript to the script that
produces it, and explains how to run them.

## Quick start

1. Install MATLAB R2025a or newer with the **Deep Learning Toolbox** and the
   **Statistics and Machine Learning Toolbox**. No GPU is needed.
2. Clone the repository and open MATLAB **in the repository root**.
3. Run the main results only (Tables 2-7 and their figures):

   ```matlab
   run_all(1)
   ```

   or everything (all tables and figures, several hours on a desktop CPU):

   ```matlab
   run_all
   ```

`run_all(stage)` runs stage 1 (main results), 2 (analysis studies of
Section 4), or 3 (appendices); `run_all([2 3])` runs several. You can also run
any single script by name, as described below.

## How the scripts behave

- **Outputs.** Each script creates `results/` if needed, saves its numbers there
  as a `.mat` file, prints its table to the command window, and writes any
  figure as a `.png` in the repository root.
- **Determinism.** Random seeds are fixed inside every script, so results are
  reproducible up to platform-level floating-point differences.
- **Data.** All data are generated analytically inside the scripts; nothing
  needs to be downloaded.
- **Resumable scripts** save after every seed and continue where they stopped
  if re-run. `algorithm2_ff.m` and `algorithm2_pinn.m` finish one case study per
  call (re-run until the summary line appears; `run_all` does this for you), and
  `rev_comparison_pinn6.m` runs one seed per call (5 calls).
- **Case-selector scripts** run one case study at a time, chosen by a variable
  set before the call, for example:

  ```matlab
  for c = 2:4, AB_CASE = c; rev_ablation5_ff; end
  ```

  | Script | Variable | Values |
  |---|---|---|
  | `rev_ablation5_ff.m` | `AB_CASE` | 2, 3, 4 |
  | `rev_ablation5_pinn.m` | `ABP_CASE` | 1 (Case Study 5), 2 (Case Study 6) |
  | `rev_optimizer_compare.m` | `ROC_CASE` | 1-4 |
  | `rev_optimizer_pinn.m` | `ROP_CASE` | 1, 2 |
  | `uq_all_ff.m` / `uq_all_pinn.m` | `UQ_CASE` / `UQP_CASE` | 1-4 / 1-2 (use `run_uq_all.m` instead) |

  Run without the variable, these scripts print the summary of what is saved.
- **Run time** (desktop CPU, MATLAB R2025b): most scripts take a few minutes;
  the multi-seed studies (ablations, uncertainty ensembles, PINN baselines,
  matched retrain) take roughly 10-60 minutes each.

## Script for every table and figure

Numbers refer to the revised manuscript. Table 1 is the nomenclature and
Figure 1 the method schematic; neither needs code.

### Main results (Section 4.1-4.6) -- `run_all(1)`

| Manuscript item | Script(s) | Output |
|---|---|---|
| Figure 2 (case-study setups) | `make_case_study_setups.m` | `case_study_setups.png` |
| Tables 2-5 (Case Studies 1-4), Table 13 rows 1-4 | `algorithm2_ff.m` (resumable) | `results/algorithm2_ff.mat` |
| Tables 6-7 (Case Studies 5-6), Table 13 rows 5-6 | `algorithm2_pinn.m` (resumable) | `results/algorithm2_pinn.mat` |
| Figures 3-6 (surrogate results) | `regen_ff_figs.m` (after `algorithm2_ff.m`) | `problem1_results.png` ... `problem4_results.png` |
| Figures 7-11 (PINN results, residual and displacement maps) | `regen_pinn_figs2.m` (after `algorithm2_pinn.m`) | `problem5_*.png`, `problem6_*.png` |
| Table 8 (physical-consistency metrics) | `rev_physics_metrics_ff.m` (surrogates); PINN residuals from `algorithm2_pinn.m` | `results/physics_metrics_ff.mat` |
| Paired unlearned-vs-retrained tests (Section 4 opening) | `rev_physics_metrics_ff.m` (Case Study 4), `rev_pinn_paired_cs6.m` (Case Study 6) | `results/physics_metrics_ff.mat`, `results/pinn_paired_cs6.mat` |
| Table 9 (wall-clock cost) | `rev_efficiency_ff.m`, `rev_efficiency_pinn.m` | `results/rev_efficiency_*.mat` |

### Analysis studies (Section 4.7-4.22) -- `run_all(2)`

| Manuscript item | Script(s) | Output |
|---|---|---|
| Figure 12 (beam uncertainty) | `run_uq_all.m` (runs `uq_all_ff.m`, `uq_all_pinn.m`, `uq_summary.m`) | `uncertainty.png`, `results/uq_all_*.mat` |
| Figure 13 (beam ablation) | `ablation_script.m` (`replot_ablation_fig.m` redraws it) | `ablation.png`, `results/ablation.mat` |
| Table 10 (beam: SISA, Fisher, retraining) | `comparison_script.m` | `results/comparison.mat` |
| Table 11 (Fisher forgetting vs Newton on the elasticity PINN) | `rev_pinn_fisher.m` | `results/rev_pinn_fisher.mat` |
| Table 12 (Adam vs Levenberg-Marquardt) | `rev_optimizer_compare.m` (`ROC_CASE`), `rev_optimizer_pinn.m` (`ROP_CASE`) | `results/rev_optimizer_*.mat` |
| Table 14 (four-way comparison), Table 15 (retain/forget/test) | `rev_fourway_ff.m`, `rev_pinn_fourway.m` | `results/rev_fourway_ff.mat`, `results/rev_pinn_fourway.mat` |
| Table 16 (PINN convergence) | `rev_pinn_convergence.m` | `results/rev_pinn_convergence.mat` |
| Figure 14 (correlated sensor drift) | `rev_correlated.m`, then `correlated_drift_fig.m` | `results/rev_correlated.mat`, `correlated_drift.png` |
| Table 17 (decomposition, beam) | `forgetting_analysis.m` | `results/forgetting.mat` |
| Table 18 (decomposition, elasticity PINN) | `pinn_decomposition.m` | printed table |
| Table 19 (Newton step, beam) | `hessian_influence.m` | `results/hessian_influence.mat` |
| Table 20 (Newton step, Neo-Hookean PINN) | `pinn_newton.m` | `results/pinn_newton.mat` |
| Damping guidance (lambda_max estimate, Section 4.18) | `damping_guidance.m` | `results/damping_guidance.mat` |
| Table 21, Figure 15 (when the Newton step helps) | `newton_necessity.m` | `newton_necessity.png`, `results/newton_necessity.mat` |
| Valid-data forgetting test (Section 4.20) | `similarity_forget.m` | `results/similarity_forget.mat` |
| Table 22 (physics weight) | `beta_ablation_5seed.m` | `results/beta_ablation_5seed.mat` |
| Table 23, Figure 16 (model-size scaling) | `scaling_analysis.m` | `scaling.png`, `results/scaling.mat` |

### Appendices -- `run_all(3)` (Appendix A is produced in stage 2)

| Manuscript item | Script(s) | Output |
|---|---|---|
| Table 24, Figure 17 (Appendix A, uncertainty across cases) | `run_uq_all.m` | `uq_appendix.png` |
| Table 25 (Appendix B, ablation across cases, 5 seeds, matched baseline) | `rev_ablation5_ff.m` (`AB_CASE`=2:4), `rev_ablation5_pinn.m` (`ABP_CASE`=1:2) | `results/ablation5_ff.mat`, `results/ablation5_pinn.mat` |
| Table 26 (Appendix C, SISA and Fisher on Case Studies 2-4 and 6) | `rev_comparison_surrogates.m`, `rev_comparison_pinn6.m` (5 calls) | `results/rev_comparison_*.mat` |
| Table 27 (Appendix D, matched-iteration retrain and safeguard check) | `rev_matched_retrain_ff.m`, `rev_matched_retrain_pinn.m` | `results/matched_retrain_*.mat` |

## Training optimizers

The main results and most studies train every model with Adam in MATLAB's
`dlarray` framework, which supplies the Hessian-vector products for the Newton
step. A few supporting beam-level studies use MATLAB's `feedforwardnet` with
Levenberg-Marquardt: the beam decomposition (`forgetting_analysis.m`), the beam
ablation (`ablation_script.m`), the SISA/Fisher comparisons on the surrogates
(`comparison_script.m`, `rev_comparison_surrogates.m`), and the valid-data test
(`similarity_forget.m`). `scaling_analysis.m` uses scaled conjugate gradient.
Each of these compares conditions trained with the same optimizer.

## PINN reference displacement field

Both PINN case studies (5 and 6) generate their boundary data and test field
from the closed-form reference field `uref`, defined and documented in
[`algorithm2_pinn.m`](algorithm2_pinn.m#L37-L53):

```
u_x =  a*x*(1 + R^2/(2 r^2)),   u_y = -b*y*(1 + R^2/(2 r^2)),   r^2 = x^2 + y^2
```

with hole radius `R = 0.5`, `(a, b) = (0.1, 0.033)` for Case Study 5 and
`(0.05, 0.05)` for Case Study 6. It is a uniform-tension field modulated by a
term that decays away from the hole. It resembles, but is not, the exact
plate-with-hole (Kirsch) solution and does not satisfy equilibrium exactly, so
part of the clean model's equilibrium residual comes from the reference itself.
All PINN models (clean, biased, unlearned, retrained) are measured against this
same field, so the comparisons between them are unaffected. Every PINN script
uses the same definition.

## Applying the method to your own data

`real_data_unlearning.m` is a data-agnostic template. Supply a CSV of
feature-target rows with an `is_removed` column marking the points to remove;
it trains the biased and retrained models, applies the damped Newton step and
the fine-tune, and reports the clean, biased, and unlearned errors. It does not
generate data and stops if no dataset is given. It is not used for any
reported result.

## Other files (not needed to reproduce the paper)

- `UQ.m` -- earlier uncertainty script. **Do not run it after `run_uq_all.m`:**
  it overwrites `uncertainty.png` with the pre-revision version.
- `computational_efficiency.m` -- collates timings from the pre-revision
  prototypes; Table 9 now comes from `rev_efficiency_*.m`.
- `replot_analysis_figs.m` -- redraws `scaling.png` from `results/scaling.mat`
  (it also writes `computational_efficiency.png`, a figure no longer in the paper).
- `ablation_all_ff.m`, `ablation_all_pinn.m` -- earlier 2-seed version of
  Appendix B, superseded by `rev_ablation5_*.m`.
- `hessian_influence_resumable.m`, `pinn_newton_resumable.m` -- resumable copies
  of `hessian_influence.m` and `pinn_newton.m` (same results, one seed per call).
- `check_beam_min.m` -- small diagnostic of the beam training data.
- `comparison_bars_fig.m` -- bar-chart version of Table 10 (from `results/comparison.mat`);
  not included in the final manuscript.
- `make_casestudy_fig.m` -- early draft of Figure 2; the published figure comes
  from `make_case_study_setups.m`.
