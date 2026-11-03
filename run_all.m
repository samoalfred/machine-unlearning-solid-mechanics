function run_all(stages)
%RUN_ALL  Regenerate every table and figure of the manuscript, in order.
%
%   run_all          runs all stages (several hours on a desktop CPU)
%   run_all(1)       runs only stage 1 (main results, Tables 2-7, Figures 3-11)
%   run_all([2 3])   runs stages 2 and 3
%
%   Stage 1  Main results: the six case studies (Tables 2-7), their figures,
%            physical-consistency metrics (Table 8), paired tests, timings (Table 9)
%   Stage 2  Analysis studies of Section 4 (uncertainty, ablation, baselines,
%            optimizer, four-way comparison, convergence, drift, decomposition,
%            Newton step, necessity, valid-data test, physics weight, scaling)
%   Stage 3  Appendices B-D (across-case ablation, baselines, matched retrain)
%
%   Run from the repository root. Every script writes its .mat output to
%   results/ and its figures (.png) to the repository root. Scripts are run in
%   the base workspace, so each behaves exactly as if typed at the prompt.
%   See REPRODUCE.md for which script produces which table or figure.

if nargin < 1, stages = 1:3; end
t0 = tic;

if any(stages == 1)
    step('Stage 1: main results');
    for i = 1:6, run1('algorithm2_ff'); end      % resumable: one case per call (Tables 2-5)
    for i = 1:3, run1('algorithm2_pinn'); end    % resumable: one case per call (Tables 6-7)
    run1('regen_ff_figs');                       % Figures 3-6
    run1('regen_pinn_figs2');                    % Figures 7-11
    run1('make_case_study_setups');              % Figure 2
    run1('rev_physics_metrics_ff');              % Table 8 (surrogates)
    run1('rev_pinn_paired_cs6');                 % paired test, Case Study 6
    run1('rev_efficiency_ff'); run1('rev_efficiency_pinn');   % Table 9
end

if any(stages == 2)
    step('Stage 2: analysis studies (Section 4)');
    run1('run_uq_all');                          % Figure 12, Table 24, Figure 17
    run1('ablation_script');                     % Figure 13 (beam ablation)
    run1('comparison_script');                   % Table 10
    run1('rev_pinn_fisher');                     % Table 11
    for c = 1:4, run1(sprintf('ROC_CASE=%d; rev_optimizer_compare', c)); end   % Table 12 (surrogates)
    for c = 1:2, run1(sprintf('ROP_CASE=%d; rev_optimizer_pinn', c)); end      % Table 12 (PINNs)
    run1('clear ROC_CASE ROP_CASE');
    run1('rev_fourway_ff'); run1('rev_pinn_fourway');          % Tables 14-15
    run1('rev_pinn_convergence');                % Table 16
    run1('rev_correlated'); run1('correlated_drift_fig');      % Figure 14
    run1('forgetting_analysis');                 % Table 17
    run1('pinn_decomposition');                  % Table 18
    run1('hessian_influence');                   % Table 19
    run1('pinn_newton');                         % Table 20
    run1('damping_guidance');                    % lambda_max estimate (Section 4.18)
    run1('newton_necessity');                    % Table 21, Figure 15
    run1('similarity_forget');                   % valid-data test (Section 4.20)
    run1('beta_ablation_5seed');                 % Table 22
    run1('scaling_analysis');                    % Table 23, Figure 16
end

if any(stages == 3)
    step('Stage 3: appendices');
    for c = 2:4, run1(sprintf('AB_CASE=%d; rev_ablation5_ff', c)); end      % Table 25 (surrogates)
    for c = 1:2, run1(sprintf('ABP_CASE=%d; rev_ablation5_pinn', c)); end   % Table 25 (PINNs)
    run1('clear AB_CASE ABP_CASE');
    run1('rev_comparison_surrogates');           % Table 26 (Case Studies 2-4)
    for i = 1:5, run1('rev_comparison_pinn6'); end   % Table 26 (Case Study 6), one seed per call
    run1('rev_matched_retrain_ff'); run1('rev_matched_retrain_pinn');    % Table 27
end

fprintf('\nrun_all finished in %.1f min. Outputs are in results/ and *.png.\n', toc(t0)/60);
end

function run1(cmd)
    fprintf('\n>> %s\n', cmd);
    evalin('base', cmd);
end

function step(msg)
    fprintf('\n==================== %s ====================\n', msg);
end
