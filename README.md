# Machine Unlearning for Neural-Network Models in Solid Mechanics

MATLAB code accompanying the manuscript *"Machine Unlearning for Neural-Network
Models in Solid Mechanics"* (Samuel Onimpa Alfred, Aerospace Engineering
Department, University of Michigan, Ann Arbor).

The code removes the influence of known faulty or obsolete training data from a
trained neural network without retraining it from scratch. It covers
feedforward surrogates and physics-informed neural networks (PINNs) on six
solid-mechanics case studies, and reproduces every number, table, and figure in
the paper.

## The method in brief

Given a trained model and the set of points to remove, Algorithm 1 in the paper:

1. **Newton step.** Solves the damped influence system
   `(H + gamma*I) v = g_removed` by conjugate gradient, using Hessian-vector
   products from automatic differentiation (the Hessian is never formed), and
   updates the weights by `v`. The damping `gamma` starts at `1e-3 * lambda_max`
   (estimated by power iteration) and is raised tenfold if a step would blow up
   the loss on the retained data.
2. **Warm-start fine-tune.** Continues training on the retained data for a short
   budget; for the PINNs the loss also includes the equilibrium residual.

The main implementations are in `algorithm2_ff.m` (surrogates) and
`algorithm2_pinn.m` (PINNs).

## Requirements

- MATLAB R2025a or newer (tested with R2025b)
- Deep Learning Toolbox (`dlarray`, `dlfeval`, `dlgradient`, `adamupdate`, `feedforwardnet`)
- Statistics and Machine Learning Toolbox (paired t-tests)
- No GPU and no external data: all data are generated inside the scripts.

## Quick start

Open MATLAB in the repository root and run:

```matlab
run_all(1)   % main results: Tables 2-7 and Figures 2-11 (minutes to about an hour)
run_all      % everything in the paper (several hours on a desktop CPU)
```

Each script saves its numbers to `results/` (created automatically), prints its
table, and writes its figures as `.png` files in the repository root. Fixed
random seeds make the results reproducible.

## Where to find things

**[REPRODUCE.md](REPRODUCE.md)** lists the script behind every table and figure,
how to run single scripts, which ones are resumable or take a case-study
setting, and which files are not needed to reproduce the paper.

| Paper section | Main scripts |
|---|---|
| Main results (Tables 2-9) | `algorithm2_ff.m`, `algorithm2_pinn.m`, `regen_ff_figs.m`, `regen_pinn_figs2.m`, `rev_physics_metrics_ff.m`, `rev_efficiency_*.m` |
| Analysis studies (Section 4) | `run_uq_all.m`, `ablation_script.m`, `comparison_script.m`, `rev_fourway_ff.m`, `forgetting_analysis.m`, `hessian_influence.m`, `newton_necessity.m`, `scaling_analysis.m`, and others |
| Appendices B-D | `rev_ablation5_*.m`, `rev_comparison_*.m`, `rev_matched_retrain_*.m` |

## Applying the method to your own data

`real_data_unlearning.m` is a data-agnostic template: supply a CSV of
feature-target rows with an `is_removed` column (1 = remove, 0 = keep), and it
trains the biased and retrained models, applies the Newton step and fine-tune,
and reports the clean, biased, and unlearned errors.

## Scope

All case studies use analytically generated data with controlled, injected
corruption, and the points to remove are assumed known. The study is a proof of
concept; validation on real experimental or finite-element data is future work.

## License

Released under the [MIT License](LICENSE).

## Citation

If you use this code, please cite the accompanying manuscript (full reference
to be added on publication).
