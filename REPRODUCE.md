# Reproducing the results

This document lists, in order, the scripts to run to reproduce every table and
figure in the manuscript. Each script is self-contained: it creates a `results/`
folder, saves its `.mat` outputs there, prints its tables to the command window,
and (where applicable) writes its `.png` figures to the working directory.

Run everything from the repository root in MATLAB (R2025a+, with the Deep
Learning and Statistics and Machine Learning toolboxes). Fixed random seeds are
set inside each script, so results are deterministic up to platform-level
floating-point differences.

## Step 0 - Primary results first (other steps do not depend on these, but the figures do)

The two "primary" scripts are **resumable**: each processes one case study (and
a few seeds) per invocation and appends to its `.mat`. Re-run each repeatedly
until it prints its final summary line.

1. **`algorithm2_ff.m`** - feedforward case studies 1-4 (10 seeds each).
   Re-run until it prints `=== Algorithm 2 feedforward summary (10 seeds) ===`.
   Produces `results/algorithm2_ff.mat` -> **Tables 1-4** and the significance rows.
2. **`algorithm2_pinn.m`** - PINN case studies 5-6 (5 seeds each).
   Re-run until it prints `=== Algorithm 2 PINN summary (5 seeds) ===`.
   Produces `results/algorithm2_pinn.mat` -> **Tables 5-6** and the significance rows.

## Step 1 - Case-study result figures (depend on Step 0)

3. **`regen_ff_figs.m`** - loads `results/algorithm2_ff.mat`; writes
   `problem1_results.png` ... `problem4_results.png` -> **Figures 2-5**.
4. **`regen_pinn_figs2.m`** - loads `results/algorithm2_pinn.mat`; writes
   `problem5_results.png`, `problem6_results.png`, `problem5_residuals.png`,
   `problem6_residuals.png`, `problem6_displacement.png` -> **Figures 6-7** and
   the residual/displacement figures.

## Step 2 - Analysis and ablation studies (independent; run in any order)

Each of these is standalone (no dependency on Steps 0-1) and reproduces one part
of the Results section:

5. **`UQ.m`** -> `uncertainty.png`, `results/uq.mat` (uncertainty quantification).
6. **`ablation_script.m`** -> `ablation.png`, `results/ablation.mat` (ablation sweeps).
7. **`comparison_script.m`** -> `results/comparison.mat` (SISA / Fisher-forgetting comparison table).
8. **`forgetting_analysis.m`** -> `results/forgetting.mat` (decomposition on the beam).
9. **`pinn_decomposition.m`** -> decomposition on the 2D elasticity PINN.
10. **`hessian_influence.m`** -> `results/hessian_influence.mat` (second-order Newton step, beam table).
11. **`pinn_newton.m`** -> `results/pinn_newton.mat` (second-order Newton step, Neo-Hookean PINN table).
12. **`damping_guidance.m`** -> `results/damping_guidance.mat` (power-iteration lambda_max estimate).
13. **`newton_necessity.m`** -> `newton_necessity.png`, `results/newton_necessity.mat`
    (8-seed budget and removal-fraction sweeps with adaptive damping; the "when the
    Newton step accelerates recovery" table and figure).
14. **`beta_ablation_5seed.m`** -> `results/beta_ablation_5seed.mat` (physics-weight sweep, 5 seeds).
15. **`similarity_forget.m`** -> `results/similarity_forget.mat` (valid-data forgetting test).
16. **`scaling_analysis.m`** -> `scaling.png`, `results/scaling.mat` (model-size scaling).

## Step 3 - Efficiency figure (depends on some Step-0/2 outputs)

17. **`rev_efficiency_ff.m`** and **`rev_efficiency_pinn.m`** -> Table 9 of the
    manuscript (`results/rev_efficiency_ff.mat`, `results/rev_efficiency_pinn.mat`).
    They time a full cold retrain against Algorithm 1 (adaptive damped Newton step +
    fine-tune) with the same budgets as the main results, 10 runs per case.
    `computational_efficiency.m` is kept for reference only: it collates timings
    from the archived pre-revision prototypes and is not used for Table 9.

## One-shot driver (optional)

To run everything non-interactively, execute the following in MATLAB from the
repo root. The two resumable scripts are looped until complete:

```matlab
% primary results (resumable): loop until all case studies are finalized
for i=1:6, run('algorithm2_ff.m');  end   % 4 case studies x seeds
for i=1:3, run('algorithm2_pinn.m'); end   % 2 case studies x seeds
% figures for the case studies
run('regen_ff_figs.m'); run('regen_pinn_figs2.m');
% analysis and ablations
for s = ["UQ","ablation_script","comparison_script","forgetting_analysis", ...
         "pinn_decomposition","hessian_influence","pinn_newton","damping_guidance", ...
         "newton_necessity","beta_ablation_5seed","similarity_forget","scaling_analysis"]
    run(s + ".m");
end
% efficiency (collates timings) last
run('computational_efficiency.m');
```

(The loop counts for the two resumable scripts are upper bounds; each script
returns early once its `.mat` is complete, so extra iterations are no-ops.)

## Not part of reproduction

- `MLU_*` / `MUL_*` - the original per-case-study prototypes, superseded by
  `algorithm2_ff.m` / `algorithm2_pinn.m` for the reported numbers.
- `beta_ablation.m` (single seed) - superseded by `beta_ablation_5seed.m`.
- `regen_pinn_figs.m` - superseded by `regen_pinn_figs2.m`.
- `real_data_unlearning.m` - drop-in template for applying the method to a real
  dataset; not used for any reported result.

## Revision experiments (major revision, round 1)

These scripts were added for the first major revision and produce the new
tables and figures requested by the reviewers:

- `rev_fourway_ff.m` -- four-way comparison (fine-tune only / Newton only /
  Newton+fine-tune / full retrain) with retain, forget, and test metrics, for
  the four feedforward case studies.
- `rev_pinn_fourway.m` -- the same four-way comparison for the two PINN cases.
- `rev_pinn_convergence.m` -- PINN convergence study (training-iteration and
  collocation-density sweeps, plus the full pipeline re-run at the largest budget).
- `rev_correlated.m` -- a spatially correlated sensor-drift corruption (a harder,
  more realistic, non-i.i.d. test).
- `rev_pinn_fisher.m` -- a stronger external baseline (diagonal-Fisher forgetting)
  on the 2D elasticity PINN.

Each is self-contained and writes its `.mat` output to `results/`.

## Revision experiments (major revision, rounds 2-5)

- `rev_optimizer_compare.m`, `rev_optimizer_pinn.m` -- sensitivity to the training
  optimizer (Adam vs Levenberg--Marquardt, tight and loss-matched).
- `rev_comparison_surrogates.m`, `rev_comparison_pinn6.m` -- SISA / Fisher
  baselines on Case Studies 2-4 and the Neo-Hookean PINN (Appendix C);
  `comparison_bars_fig.m` draws the comparison figure.
- `correlated_drift_fig.m` -- figure for the correlated sensor-drift study
  (data from `rev_correlated.m`).
- `uq_all_ff.m`, `uq_all_pinn.m`, `uq_summary.m` -- uncertainty quantification
  across all six case studies (Appendix A). Each ensemble uses one fixed test set
  shared by all members. Run everything with **`run_uq_all.m`**.
- `ablation_all_ff.m`, `ablation_all_pinn.m` -- ablation across the case studies
  (Appendix B); `replot_ablation_fig.m`, `replot_analysis_figs.m` redraw figures.
- `rev_efficiency_ff.m`, `rev_efficiency_pinn.m` -- Table 9 timings (see Step 3).
- `rev_physics_metrics_ff.m` -- Table 8 physical-consistency metrics for Case Studies
  1, 3, 4, computed on the Algorithm 1 models (same seeds as `algorithm2_ff.m`;
  `results/physics_metrics_ff.mat`, which also stores the per-seed CS4 errors).
- `rev_pinn_paired_cs6.m` -- per-seed Case Study 6 errors (same seeds as
  `algorithm2_pinn.m`) for the paired unlearned-vs-retrained test in Section 4
  (`results/pinn_paired_cs6.mat`).
- `hessian_influence_resumable.m`, `pinn_newton_resumable.m` -- resumable copies of
  the two second-order diagnostics (identical results, one seed per call).
- `check_beam_min.m`, `make_casestudy_fig.m` -- small diagnostic / draft-figure
  helpers; `make_casestudy_fig.m` is an early draft; the published Figure 2
  (case-study schematic) is produced by **`make_case_study_setups.m`**.

Optimizer note: the primary results use Adam in the `dlarray` framework. The
decomposition, the beam ablation (`ablation_script.m`), the surrogate
SISA/Fisher comparisons, and the valid-data test use MATLAB's `feedforwardnet`
(Levenberg--Marquardt); `scaling_analysis.m` uses scaled conjugate gradient.

The beam uncertainty figure and Section 4.9 numbers now come from the corrected
Adam ensembles (`uq_all_ff.m` + `uq_summary.m`); `UQ.m` is kept for reference only.
