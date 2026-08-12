%% Problem 1: Beam Deflection with Faulty Strain Gauge Data
% Enhanced with uncertainty quantification and ablation studies

clear; clc; close all;

% Create figures folder if it doesn't exist
if ~exist('figures', 'dir')
    mkdir('figures');
end

%% Configuration
rng(42);
num_runs = 10; % For ensemble uncertainty
k = 100;
F_min = 0; F_max = 500;
N_clean = 1000;
N_bad = 50;

% Storage for results
mse_clean_all = zeros(num_runs, 1);
mse_full_all = zeros(num_runs, 1);
mse_unlearned_all = zeros(num_runs, 1);
time_retrain_all = zeros(num_runs, 1);
time_unlearn_all = zeros(num_runs, 1);
epochs_retrain_all = zeros(num_runs, 1);
epochs_unlearn_all = zeros(num_runs, 1);

fprintf('=== Problem 1: Beam Deflection ===\n');

for run = 1:num_runs
    rng(run * 42);
    
    %% Generate data
    F_clean = (F_max - F_min) * rand(N_clean, 1) + F_min;
    delta_true = F_clean / k;
    delta_clean = delta_true + 0.01 * randn(N_clean, 1);
    X_clean = F_clean';  % Row vector for MATLAB network
    T_clean = delta_clean';
    
    F_bad = (F_max - F_min) * rand(N_bad, 1) + F_min;
    delta_bad = 2 * (F_bad / k);
    X_bad = F_bad';
    T_bad = delta_bad';
    
    X_full = [X_clean, X_bad];
    T_full = [T_clean, T_bad];
    
    %% Train clean model
    tic;
    net_clean = feedforwardnet([20, 15]);
    net_clean.trainParam.epochs = 2000;
    net_clean.trainParam.goal = 1e-8;
    net_clean.trainParam.showWindow = false;
    net_clean = configure(net_clean, X_clean, T_clean);
    [net_clean, tr_clean] = train(net_clean, X_clean, T_clean);
    y_clean = net_clean(X_clean);
    mse_clean_all(run) = mean((T_clean - y_clean).^2);
    time_retrain_all(run) = toc;
    epochs_retrain_all(run) = length(tr_clean.epoch);
    
    %% Train biased model
    net_full = feedforwardnet([20, 15]);
    net_full.trainParam.epochs = 2000;
    net_full.trainParam.goal = 1e-8;
    net_full.trainParam.showWindow = false;
    net_full = configure(net_full, X_full, T_full);
    [net_full, tr_full] = train(net_full, X_full, T_full);
    y_full = net_full(X_clean);
    mse_full_all(run) = mean((T_clean - y_full).^2);
    
    %% Unlearning
    tic;
    % Extract weights
    W1 = net_full.IW{1,1}; b1 = net_full.b{1};
    W2 = net_full.LW{2,1}; b2 = net_full.b{2};
    W3 = net_full.LW{3,2}; b3 = net_full.b{3};
    
    grad_W1 = zeros(size(W1)); grad_b1 = zeros(size(b1));
    grad_W2 = zeros(size(W2)); grad_b2 = zeros(size(b2));
    grad_W3 = zeros(size(W3)); grad_b3 = zeros(size(b3));
    
    alpha = 1e-6;
    
    for i = 1:N_bad
        x_i = X_bad(i);
        t_i = T_bad(i);
        
        % Forward pass
        z1 = W1 * x_i + b1;
        a1 = tanh(z1);
        z2 = W2 * a1 + b2;
        a2 = tanh(z2);
        y_pred = W3 * a2 + b3;
        
        % Backpropagation
        dL_dy = 2 * (y_pred - t_i);
        dL_dW3 = dL_dy * a2';
        dL_db3 = dL_dy;
        dL_da2 = W3' * dL_dy;
        dL_dz2 = dL_da2 .* (1 - a2.^2);
        dL_dW2 = dL_dz2 * a1';
        dL_db2 = dL_dz2;
        dL_da1 = W2' * dL_dz2;
        dL_dz1 = dL_da1 .* (1 - a1.^2);
        dL_dW1 = dL_dz1 * x_i';
        dL_db1 = dL_dz1;
        
        grad_W1 = grad_W1 + dL_dW1;
        grad_b1 = grad_b1 + dL_db1;
        grad_W2 = grad_W2 + dL_dW2;
        grad_b2 = grad_b2 + dL_db2;
        grad_W3 = grad_W3 + dL_dW3;
        grad_b3 = grad_b3 + dL_db3;
    end
    
    W1_unlearned = W1 - alpha * grad_W1;
    b1_unlearned = b1 - alpha * grad_b1;
    W2_unlearned = W2 - alpha * grad_W2;
    b2_unlearned = b2 - alpha * grad_b2;
    W3_unlearned = W3 - alpha * grad_W3;
    b3_unlearned = b3 - alpha * grad_b3;
    
    net_unlearned = feedforwardnet([20, 15]);
    net_unlearned = configure(net_unlearned, X_clean, T_clean);
    net_unlearned.IW{1,1} = W1_unlearned;
    net_unlearned.b{1} = b1_unlearned;
    net_unlearned.LW{2,1} = W2_unlearned;
    net_unlearned.b{2} = b2_unlearned;
    net_unlearned.LW{3,2} = W3_unlearned;
    net_unlearned.b{3} = b3_unlearned;
    
    net_unlearned.trainParam.epochs = 50;
    net_unlearned.trainParam.goal = 1e-8;
    net_unlearned.trainParam.showWindow = false;
    [net_unlearned, tr_unlearn] = train(net_unlearned, X_clean, T_clean);
    
    y_unlearned = net_unlearned(X_clean);
    mse_unlearned_all(run) = mean((T_clean - y_unlearned).^2);
    time_unlearn_all(run) = toc;
    epochs_unlearn_all(run) = length(tr_unlearn.epoch);
end

%% Results
fprintf('\nResults (mean +/- std, %d runs):\n', num_runs);
fprintf('Clean Model MSE: %.6f +/- %.6f\n', mean(mse_clean_all), std(mse_clean_all));
fprintf('Biased Model MSE: %.6f +/- %.6f\n', mean(mse_full_all), std(mse_full_all));
fprintf('Unlearned Model MSE: %.6f +/- %.6f\n', mean(mse_unlearned_all), std(mse_unlearned_all));
fprintf('Retraining Time: %.2f +/- %.2f s\n', mean(time_retrain_all), std(time_retrain_all));
fprintf('Unlearning Time: %.2f +/- %.2f s\n', mean(time_unlearn_all), std(time_unlearn_all));
fprintf('Retraining Epochs: %.0f +/- %.0f\n', mean(epochs_retrain_all), std(epochs_retrain_all));
fprintf('Unlearning Epochs: %.0f +/- %.0f\n', mean(epochs_unlearn_all), std(epochs_unlearn_all));
fprintf('Speedup: %.1f +/- %.1f x\n', mean(time_retrain_all./time_unlearn_all), std(time_retrain_all./time_unlearn_all));

%% Physics-consistency metrics (computed on the last run's models)
% Constitutive residual R1 = mean( (f(F) - F/k)^2 ) and monotonicity metric
% M = mean( max(0, -d f / d F) ) evaluated by finite differences on a test grid.
F_grid = linspace(F_min, F_max, 400);
dF = F_grid(2) - F_grid(1);
true_grid = F_grid / k;

y_biased_grid    = net_full(F_grid);
y_unlearned_grid = net_unlearned(F_grid);
y_clean_grid     = net_clean(F_grid);

R1_biased    = mean((y_biased_grid    - true_grid).^2);
R1_unlearned = mean((y_unlearned_grid - true_grid).^2);
R1_clean     = mean((y_clean_grid     - true_grid).^2);

dydF_biased    = diff(y_biased_grid)    / dF;
dydF_unlearned = diff(y_unlearned_grid) / dF;
dydF_clean     = diff(y_clean_grid)     / dF;
M_biased    = mean(max(0, -dydF_biased));
M_unlearned = mean(max(0, -dydF_unlearned));
M_clean     = mean(max(0, -dydF_clean));

fprintf('\nPhysics-consistency (last run):\n');
fprintf('Constitutive residual R1  clean/biased/unlearned: %.3e / %.3e / %.3e\n', ...
    R1_clean, R1_biased, R1_unlearned);
fprintf('Monotonicity metric M     clean/biased/unlearned: %.3e / %.3e / %.3e\n', ...
    M_clean, M_biased, M_unlearned);

%% Paired significance test (clean-run reference vs biased vs unlearned)
[~, p_bias]  = ttest(mse_full_all,      mse_unlearned_all);
d_bias = (mean(mse_full_all) - mean(mse_unlearned_all)) / ...
         std(mse_full_all - mse_unlearned_all);

%% Save results for aggregation
if ~exist('results','dir'); mkdir('results'); end
res = struct();
res.name          = 'Beam Deflection';
res.mse_clean     = [mean(mse_clean_all),     std(mse_clean_all)];
res.mse_biased    = [mean(mse_full_all),      std(mse_full_all)];
res.mse_unlearned = [mean(mse_unlearned_all), std(mse_unlearned_all)];
res.err_reduction = (mean(mse_full_all) - mean(mse_unlearned_all)) / mean(mse_full_all) * 100;
res.time_retrain  = [mean(time_retrain_all),  std(time_retrain_all)];
res.time_unlearn  = [mean(time_unlearn_all),  std(time_unlearn_all)];
res.speedup       = [mean(time_retrain_all./time_unlearn_all), std(time_retrain_all./time_unlearn_all)];
res.R1            = [R1_clean, R1_biased, R1_unlearned];
res.M             = [M_clean, M_biased, M_unlearned];
res.p_value       = p_bias;
res.cohen_d       = d_bias;
res.mse_biased_all    = mse_full_all;
res.mse_unlearned_all = mse_unlearned_all;
save(fullfile('results','problem1.mat'), 'res');
fprintf('Saved results/problem1.mat\n');

%% Figure generation
figure(1); clf; set(gcf, 'Position', [50, 50, 1200, 800]);

% Create test data as ROW vector for network input
F_test = linspace(0, 500, 200);  % Row vector
F_test_col = F_test';  % Column for plotting

% Get predictions (networks expect row inputs)
y_clean_test = net_clean(F_test);
y_full_test = net_full(F_test);
y_unlearned_test = net_unlearned(F_test);

% Subplot 1: Clean model
subplot(2,2,1);
plot(F_test_col, F_test_col/k, 'k-', 'LineWidth', 2); hold on;
plot(F_test_col, y_clean_test, 'b--', 'LineWidth', 2);
scatter(F_clean, delta_clean, 10, 'b.', 'MarkerFaceAlpha', 0.5);
xlabel('Force F (N)'); ylabel('Deflection δ');
title('Clean Model');
legend('True', 'Predicted', 'Data', 'Location', 'best');
grid on;

% Subplot 2: Biased model
subplot(2,2,2);
plot(F_test_col, F_test_col/k, 'k-', 'LineWidth', 2); hold on;
plot(F_test_col, y_full_test, 'r--', 'LineWidth', 2);
scatter(F_clean, delta_clean, 10, 'b.', 'MarkerFaceAlpha', 0.5);
scatter(F_bad, delta_bad, 50, 'rx', 'LineWidth', 2);
xlabel('Force F (N)'); ylabel('Deflection δ');
title('Biased Model (With Corrupted Data)');
legend('True', 'Predicted', 'Clean Data', 'Corrupted Data', 'Location', 'best');
grid on;

% Subplot 3: Unlearned model
subplot(2,2,3);
plot(F_test_col, F_test_col/k, 'k-', 'LineWidth', 2); hold on;
plot(F_test_col, y_unlearned_test, 'm--', 'LineWidth', 2);
scatter(F_clean, delta_clean, 10, 'b.', 'MarkerFaceAlpha', 0.5);
scatter(F_bad, delta_bad, 50, 'rx', 'LineWidth', 2);
xlabel('Force F (N)'); ylabel('Deflection δ');
title('Unlearned Model');
legend('True', 'Predicted', 'Clean Data', 'Corrupted Data', 'Location', 'best');
grid on;

% Subplot 4: Error comparison with confidence intervals
subplot(2,2,4);
errors = [mean(mse_clean_all), mean(mse_full_all), mean(mse_unlearned_all)];
err_std = [std(mse_clean_all), std(mse_full_all), std(mse_unlearned_all)];
bar(errors); hold on;
errorbar(1:3, errors, err_std, 'k.', 'LineWidth', 2);
set(gca, 'XTickLabel', {'Clean', 'Biased', 'Unlearned'});
ylabel('Mean Squared Error');
title('Model Performance Comparison');
grid on;

sgtitle('Problem 1: Beam Deflection Results', 'FontSize', 14);

% Save figure
saveas(gcf, 'problem1_results.png');
fprintf('\nFigure saved to problem1_results.png\n');

%% Speedup comparison
figure(2); clf; set(gcf, 'Position', [50, 50, 600, 400]);
bar_data = [mean(time_retrain_all), mean(time_unlearn_all)];
bar(bar_data);
hold on;
errorbar(1:2, bar_data, [std(time_retrain_all), std(time_unlearn_all)], 'k.', 'LineWidth', 2);
set(gca, 'XTickLabel', {'Retraining', 'Unlearning'});
ylabel('Time (seconds)');
title('Computational Time Comparison');
grid on;
saveas(gcf, 'problem1_speedup.png');
fprintf('Figure saved to problem1_speedup.png\n');