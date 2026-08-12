# Machine Unlearning for Neural-Network Models in Solid Mechanics

MATLAB code accompanying the manuscript *"Machine Unlearning for Neural-Network
Models in Solid Mechanics"* (Samuel Onimpa Alfred, Aerospace Engineering
Department, University of Michigan, Ann Arbor).

The scripts implement a physics-informed data-removal (machine unlearning)
procedure for neural-network surrogates and physics-informed neural networks
(PINNs), and reproduce all reported numbers, tables, and figures. Unlearning is
performed by a damped second-order (Newton) influence update solved by
conjugate gradient with automatic-differentiation Hessian-vector products,
followed by a physics-regularized warm-start fine-tune.

## Requirements
- MATLAB R2025a or newer
- Deep Learning Toolbox (`dlarray`, `dlfeval`, `dlgradient`, `adamupdate`)
- Statistics and Machine Learning Toolbox (paired t-tests, effect sizes)

Scripts write their outputs to a `results/` folder (created automatically) and
save figures as `.png` in the working directory.

## Reproducibility

See **[REPRODUCE.md](REPRODUCE.md)** for the exact order in which to run the
scripts to regenerate every table and figure in the paper, including which
scripts are resumable, which figures depend on which results, and a copy-paste
one-shot driver.

## Primary results (Case Studies 1-6; Tables 1-6, Figures 2-7)
| Script | Produces |
|---|---|
| `algorithm2_ff.m` | Feedforward case studies 1-4 (10 seeds): tables + significance |
| `algorithm2_pinn.m` | PINN case studies 5-6 (5 seeds): tables + significance |
| `regen_ff_figs.m` | Four-panel result figures for case studies 1-4 |
| `regen_pinn_figs2.m` | PINN result figures, equilibrium-residual maps, displacement field |

## Analysis and ablation studies
| Script | Produces |
|---|---|
| `UQ.m` | Ensemble uncertainty quantification |
| `ablation_script.m` | Ablation sweeps (removal count, corruption magnitude, depth, step size) |
| `comparison_script.m` | Comparison with SISA and Fisher forgetting |
| `forgetting_analysis.m` | Decomposition of the recovery on the beam case study |
| `pinn_decomposition.m` | Decomposition on the 2D elasticity PINN |
| `hessian_influence.m` | Second-order (Newton) influence step on the beam |
| `pinn_newton.m` | Second-order (Newton) influence step on the Neo-Hookean PINN |
| `damping_guidance.m` | Power-iteration estimate of the Hessian spectrum for damping |
| `newton_necessity.m` | Regimes where the Newton step accelerates recovery (adaptive damping) |
| `beta_ablation_5seed.m` | Physics-weight (beta) sweep, 5 seeds |
| `similarity_forget.m` | Valid-data (obsolete-state) forgetting test |
| `scaling_analysis.m` | Model-size scaling of the warm-start speedup |
| `computational_efficiency.m` | Wall-clock efficiency comparison |

## Applying the method to your own data
`real_data_unlearning.m` is a data-agnostic template: supply a CSV of
feature-target rows with an `is_removed` flag, and it runs the full pipeline
(train biased and retrained models, damped Newton influence step, fine-tune)
and reports clean/biased/unlearned errors. It does not generate data and will
stop if no dataset is provided.

## Notes
- The six `MLU_*`/`MUL_*` example scripts are the original per-case-study
  implementations. The reported Tables 1-6 use the unified `algorithm2_ff.m` /
  `algorithm2_pinn.m` runs, which supersede them.
- `beta_ablation.m` (single seed) is superseded by `beta_ablation_5seed.m`;
  `regen_pinn_figs.m` is superseded by `regen_pinn_figs2.m`.
- All data are analytically generated with controlled corruption; this study is
  a proof of concept, and validation on real experimental or finite-element
  datasets is future work.
