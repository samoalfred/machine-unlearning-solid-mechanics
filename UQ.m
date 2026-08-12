%% Uncertainty Quantification for Unlearning
% Ensemble method for quantifying uncertainty

clear; clc; close all;

% Create figures folder if it doesn't exist
if ~exist('figures', 'dir')
    mkdir('figures');
end

rng(42);

%% Setup
k = 100;
F_min = 0; F_max = 500;
N_clean = 1000;
N_bad = 50;
M = 10; % Ensemble size

% Generate data
F_clean = (F_max - F_min) * rand(N_clean, 1) + F_min;
delta_clean = F_clean / k + 0.01 * randn(N_clean, 1);
X_clean = F_clean';
T_clean = delta_clean';

F_bad = (F_max - F_min) * rand(N_bad, 1) + F_min;
delta_bad = 2 * (F_bad / k);
X_bad = F_bad';
T_bad = delta_bad';

X_full = [X_clean, X_bad];
T_full = [T_clean, T_bad];

%% Train ensemble of biased models
fprintf('Training ensemble of %d biased models...\n', M);

y_pred_full = zeros(M, N_clean);
networks_full = cell(M, 1);

for m = 1:M
    rng(m * 1000 + 42);
    net = feedforwardnet([20, 15]);
    net.trainParam.epochs = 1000;
    net.trainParam.goal = 1e-8;
    net.trainParam.showWindow = false;
    net = configure(net, X_full, T_full);
    net = train(net, X_full, T_full);
    networks_full{m} = net;
    y_pred_full(m, :) = net(X_clean);
end

% Compute mean and variance
mu_full = mean(y_pred_full, 1);
sigma_full = std(y_pred_full, 0, 1);

%% Train ensemble of unlearned models
fprintf('Training ensemble of %d unlearned models...\n', M);

y_pred_unlearned = zeros(M, N_clean);

for m = 1:M
    % Extract weights from biased model
    net_full = networks_full{m};
    W1 = net_full.IW{1,1}; b1 = net_full.b{1};
    W2 = net_full.LW{2,1}; b2 = net_full.b{2};
    W3 = net_full.LW{3,2}; b3 = net_full.b{3};
    
    % Compute gradients and unlearn
    grad_W1 = zeros(size(W1)); grad_b1 = zeros(size(b1));
    grad_W2 = zeros(size(W2)); grad_b2 = zeros(size(b2));
    grad_W3 = zeros(size(W3)); grad_b3 = zeros(size(b3));
    alpha = 1e-6;
    
    for i = 1:N_bad
        x_i = X_bad(i);
        t_i = T_bad(i);
        
        z1 = W1 * x_i + b1;
        a1 = tanh(z1);
        z2 = W2 * a1 + b2;
        a2 = tanh(z2);
        y_pred = W3 * a2 + b3;
        
        dL_dy = 2 * (y_pred - t_i);
        dL_dW3 = dL_dy * a2'; dL_db3 = dL_dy;
        dL_da2 = W3' * dL_dy; dL_dz2 = dL_da2 .* (1 - a2.^2);
        dL_dW2 = dL_dz2 * a1'; dL_db2 = dL_dz2;
        dL_da1 = W2' * dL_dz2; dL_dz1 = dL_da1 .* (1 - a1.^2);
        dL_dW1 = dL_dz1 * x_i'; dL_db1 = dL_dz1;
        
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
    
    net_unlearned.trainParam.epochs = 30;
    net_unlearned.trainParam.goal = 1e-8;
    net_unlearned.trainParam.showWindow = false;
    net_unlearned = train(net_unlearned, X_clean, T_clean);
    
    y_pred_unlearned(m, :) = net_unlearned(X_clean);
end

mu_unlearned = mean(y_pred_unlearned, 1);
sigma_unlearned = std(y_pred_unlearned, 0, 1);

%% Train ensemble of CLEAN models (reference epistemic uncertainty)
fprintf('Training ensemble of %d clean models...\n', M);
y_pred_clean = zeros(M, N_clean);
for m = 1:M
    rng(m*2000 + 7);
    net = feedforwardnet([20, 15]);
    net.trainParam.epochs = 1000; net.trainParam.goal = 1e-8; net.trainParam.showWindow = false;
    net = configure(net, X_clean, T_clean); net = train(net, X_clean, T_clean);
    y_pred_clean(m, :) = net(X_clean);
end
mu_clean = mean(y_pred_clean, 1);
sigma_clean = std(y_pred_clean, 0, 1);

%% Real UQ metrics
delta_true_clean = F_clean' / k;                  % true physics at training inputs
region = F_clean' > 400;                          % region influenced by corrupted data

% 95% prediction-interval coverage (epistemic +/- 2 sigma) for the unlearned ensemble
cover_unlearned = mean(abs(mu_unlearned - delta_true_clean) <= 2*sigma_unlearned) * 100;
cover_clean     = mean(abs(mu_clean     - delta_true_clean) <= 2*sigma_clean)     * 100;

% Epistemic uncertainty increase of the biased model in the corrupted region
unc_increase = (mean(sigma_full(region)) - mean(sigma_clean(region))) / mean(sigma_clean(region)) * 100;
unc_recovery = (mean(sigma_unlearned(region)) - mean(sigma_clean(region))) / mean(sigma_clean(region)) * 100;

fprintf('\n--- Uncertainty quantification (computed) ---\n');
fprintf('95%% PI coverage: clean %.1f%%, unlearned %.1f%%\n', cover_clean, cover_unlearned);
fprintf('Epistemic uncertainty increase (biased vs clean, F>400): %.0f%%\n', unc_increase);
fprintf('Residual uncertainty after unlearning (vs clean, F>400): %.0f%%\n', unc_recovery);

if ~exist('results','dir'); mkdir('results'); end
uq = struct('M',M,'coverage_unlearned',cover_unlearned,'coverage_clean',cover_clean, ...
    'unc_increase_biased',unc_increase,'unc_residual_unlearned',unc_recovery);
save(fullfile('results','uq.mat'),'uq');

%% Visualize uncertainty
figure(1); clf; set(gcf, 'Position', [50, 50, 1200, 400]);

F_test = linspace(F_min, F_max, 200)';
delta_true = F_test / k;

% Sort for plotting - ensure all arrays are column vectors
[F_sorted, idx] = sort(F_clean);
F_sorted = F_sorted(:);
mu_full_sorted = mu_full(idx)';
sigma_full_sorted = sigma_full(idx)';
mu_unlearned_sorted = mu_unlearned(idx)';
sigma_unlearned_sorted = sigma_unlearned(idx)';

subplot(1,3,1);
plot(F_test, delta_true, 'k-', 'LineWidth', 2); hold on;
plot(F_sorted, mu_full_sorted, 'b-', 'LineWidth', 2);
% Fill with proper vectors
fill([F_sorted; flipud(F_sorted)], ...
     [mu_full_sorted + 2*sigma_full_sorted; flipud(mu_full_sorted - 2*sigma_full_sorted)], ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
xlabel('Force F (N)'); ylabel('Deflection δ');
title('Biased Model with Uncertainty');
legend('True', 'Mean', '95% CI', 'Location', 'best');
grid on;

subplot(1,3,2);
plot(F_test, delta_true, 'k-', 'LineWidth', 2); hold on;
plot(F_sorted, mu_unlearned_sorted, 'm-', 'LineWidth', 2);
fill([F_sorted; flipud(F_sorted)], ...
     [mu_unlearned_sorted + 2*sigma_unlearned_sorted; flipud(mu_unlearned_sorted - 2*sigma_unlearned_sorted)], ...
     'm', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
xlabel('Force F (N)'); ylabel('Deflection δ');
title('Unlearned Model with Uncertainty');
legend('True', 'Mean', '95% CI', 'Location', 'best');
grid on;

subplot(1,3,3);
plot(F_sorted, sigma_full_sorted, 'b-', 'LineWidth', 2); hold on;
plot(F_sorted, sigma_unlearned_sorted, 'm-', 'LineWidth', 2);
scatter(F_bad, zeros(size(F_bad)) + 0.5*max(sigma_full_sorted), 50, 'rx', 'LineWidth', 2);
xlabel('Force F (N)'); ylabel('Uncertainty (std)');
title('Uncertainty Comparison');
legend('Biased', 'Unlearned', 'Corrupted Data', 'Location', 'best');
grid on;

sgtitle('Uncertainty Quantification', 'FontSize', 14);

exportgraphics(gcf, 'uncertainty.png', 'Resolution', 200);
fprintf('Figure saved to uncertainty.png (200 dpi)\n');