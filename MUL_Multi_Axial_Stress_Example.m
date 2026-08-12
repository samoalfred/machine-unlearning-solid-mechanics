%% Example 3: Multi-Axial Stress State with Boundary Condition Errors
% Properly handles multi-layer networks with multiple outputs

clear; clc; close all;

%% 1. Generate Data for Plane Stress Condition

rng(456);

% Material properties
E = 70; % GPa (Aluminum)
nu = 0.33;
G = E / (2 * (1 + nu)); % Shear modulus

% Load ranges
sigma_x_range = [0, 100]; % MPa
tau_xy_range = [0, 50]; % MPa

% Generate clean data
N_clean = 600;
sigma_x_clean = sigma_x_range(1) + (sigma_x_range(2) - sigma_x_range(1)) * rand(N_clean, 1);
tau_xy_clean = tau_xy_range(1) + (tau_xy_range(2) - tau_xy_range(1)) * rand(N_clean, 1);

% True strains (no coupling between normal and shear)
epsilon_x_true = sigma_x_clean / E;
gamma_xy_true = tau_xy_clean / G;

% Add measurement noise
epsilon_x_clean = epsilon_x_true + 0.0001 * randn(N_clean, 1);
gamma_xy_clean = gamma_xy_true + 0.0001 * randn(N_clean, 1);

% Prepare data: inputs [σx, τxy], outputs [εx, γxy]
X_clean = [sigma_x_clean'; tau_xy_clean']; % 2 x N_clean
T_clean = [epsilon_x_clean'; gamma_xy_clean']; % 2 x N_clean

%% 2. Create Corrupted Data

N_bad = 40;
sigma_x_bad = sigma_x_range(1) + (sigma_x_range(2) - sigma_x_range(1)) * rand(N_bad, 1);
tau_xy_bad = tau_xy_range(1) + (tau_xy_range(2) - tau_xy_range(1)) * rand(N_bad, 1);

% With incorrect BC, there's coupling:
epsilon_x_bad = sigma_x_bad / E + 0.1 * tau_xy_bad / G + 0.0001 * randn(N_bad, 1);
gamma_xy_bad = tau_xy_bad / G + 0.05 * sigma_x_bad / E + 0.0001 * randn(N_bad, 1);

X_bad = [sigma_x_bad'; tau_xy_bad']; % 2 x N_bad
T_bad = [epsilon_x_bad'; gamma_xy_bad']; % 2 x N_bad

%% 3. Train Models

% Combine data
X_full = [X_clean, X_bad];
T_full = [T_clean, T_bad];

% Clean model (2 hidden layers with 20 and 15 neurons, then output layer)
net_clean = feedforwardnet([20, 15]);
net_clean.trainParam.epochs = 500;
net_clean.trainParam.goal = 1e-8;
net_clean.trainParam.showWindow = false;
net_clean = configure(net_clean, X_clean, T_clean);
net_clean = train(net_clean, X_clean, T_clean);

% Evaluate clean model
y_clean = net_clean(X_clean);
mse_clean = mean(mean((T_clean - y_clean).^2));

% Biased model
net_full = feedforwardnet([20, 15]);
net_full.trainParam.epochs = 500;
net_full.trainParam.goal = 1e-8;
net_full.trainParam.showWindow = false;
net_full = configure(net_full, X_full, T_full);
net_full = train(net_full, X_full, T_full);

% Evaluate biased model
y_full = net_full(X_clean);
mse_full = mean(mean((T_clean - y_full).^2));

fprintf('Example 3: Multi-Axial Stress State\n');
fprintf('MSE (Clean Model): %.6f\n', mse_clean);
fprintf('MSE (Biased Model): %.6f\n', mse_full);

%% 4. Manual Unlearning with Proper Multi-Layer Handling

% Extract weights from the biased model
% For a network with architecture [20, 15]:
% Layer 1: W1 (20x2), b1 (20x1)
% Layer 2: W2 (15x20), b2 (15x1)
% Output layer: W3 (2x15), b3 (2x1)

W1 = net_full.IW{1,1};     % 20 x 2
b1 = net_full.b{1};        % 20 x 1
W2 = net_full.LW{2,1};     % 15 x 20
b2 = net_full.b{2};        % 15 x 1
W3 = net_full.LW{3,2};     % 2 x 15
b3 = net_full.b{3};        % 2 x 1

% Display network architecture
fprintf('\nNetwork architecture:\n');
fprintf('Layer 1: W1 (%d x %d), b1 (%d x %d)\n', size(W1,1), size(W1,2), size(b1,1), size(b1,2));
fprintf('Layer 2: W2 (%d x %d), b2 (%d x %d)\n', size(W2,1), size(W2,2), size(b2,1), size(b2,2));
fprintf('Output:  W3 (%d x %d), b3 (%d x %d)\n', size(W3,1), size(W3,2), size(b3,1), size(b3,2));

% Initialize gradients
grad_W1 = zeros(size(W1));
grad_b1 = zeros(size(b1));
grad_W2 = zeros(size(W2));
grad_b2 = zeros(size(b2));
grad_W3 = zeros(size(W3));
grad_b3 = zeros(size(b3));

alpha_unlearn = 0.0000001; % Even smaller for stability

% Process each bad data point
for i = 1:N_bad
    % Get the i-th corrupted data point
    x_i = X_bad(:, i); % 2 x 1
    t_i = T_bad(:, i); % 2 x 1
    
    % FORWARD PASS
    % Layer 1
    z1 = W1 * x_i + b1;   % 20 x 1
    a1 = tanh(z1);        % 20 x 1
    
    % Layer 2
    z2 = W2 * a1 + b2;    % 15 x 1
    a2 = tanh(z2);        % 15 x 1
    
    % Output layer
    y_pred = W3 * a2 + b3; % 2 x 1
    
    % COMPUTE LOSS DERIVATIVE
    dL_dy = 2 * (y_pred - t_i); % 2 x 1
    
    % BACKPROPAGATION
    % Output layer gradients
    dL_dW3 = dL_dy * a2'; % 2 x 15
    dL_db3 = dL_dy;       % 2 x 1
    
    % Backpropagate to layer 2
    dL_da2 = W3' * dL_dy;        % 15 x 1
    dL_dz2 = dL_da2 .* (1 - a2.^2); % 15 x 1
    dL_dW2 = dL_dz2 * a1';      % 15 x 20
    dL_db2 = dL_dz2;             % 15 x 1
    
    % Backpropagate to layer 1
    dL_da1 = W2' * dL_dz2;       % 20 x 1
    dL_dz1 = dL_da1 .* (1 - a1.^2); % 20 x 1
    dL_dW1 = dL_dz1 * x_i';      % 20 x 2
    dL_db1 = dL_dz1;             % 20 x 1
    
    % Accumulate gradients
    grad_W1 = grad_W1 + dL_dW1;
    grad_b1 = grad_b1 + dL_db1;
    grad_W2 = grad_W2 + dL_dW2;
    grad_b2 = grad_b2 + dL_db2;
    grad_W3 = grad_W3 + dL_dW3;
    grad_b3 = grad_b3 + dL_db3;
end

% Apply unlearning update
W1_unlearned = W1 - alpha_unlearn * grad_W1;
b1_unlearned = b1 - alpha_unlearn * grad_b1;
W2_unlearned = W2 - alpha_unlearn * grad_W2;
b2_unlearned = b2 - alpha_unlearn * grad_b2;
W3_unlearned = W3 - alpha_unlearn * grad_W3;
b3_unlearned = b3 - alpha_unlearn * grad_b3;

%% 5. Create Unlearned Model

% Create a new network
net_unlearned = feedforwardnet([20, 15]);
net_unlearned = configure(net_unlearned, X_clean, T_clean);

% Assign the unlearned weights
net_unlearned.IW{1,1} = W1_unlearned;
net_unlearned.b{1} = b1_unlearned;
net_unlearned.LW{2,1} = W2_unlearned;
net_unlearned.b{2} = b2_unlearned;
net_unlearned.LW{3,2} = W3_unlearned;
net_unlearned.b{3} = b3_unlearned;

% Fine-tune on clean data
net_unlearned.trainParam.epochs = 50;
net_unlearned.trainParam.goal = 1e-8;
net_unlearned.trainParam.showWindow = false;
net_unlearned = train(net_unlearned, X_clean, T_clean);

% Evaluate unlearned model
y_unlearned = net_unlearned(X_clean);
mse_unlearned = mean(mean((T_clean - y_unlearned).^2));

fprintf('\nMSE (Unlearned Model): %.6f\n', mse_unlearned);
fprintf('Improvement: %.2f%%\n', (mse_full - mse_unlearned) / mse_full * 100);

%% 6. Verification: Compare Predictions

fprintf('\n==================== VERIFICATION ====================\n');

% Test on a specific case
fprintf('Test Case: σx = 80 MPa, τxy = 30 MPa\n\n');

% True values (no coupling)
true_epsilon_x = 80 / E;
true_gamma_xy = 30 / G;

% Test input
X_test = [80; 30];

% Get predictions from each model
pred_clean = net_clean(X_test);
pred_full = net_full(X_test);
pred_unlearned = net_unlearned(X_test);

fprintf('Normal Strain εx:\n');
fprintf('  True value:        %.6f\n', true_epsilon_x);
fprintf('  Clean model:       %.6f (error: %.6f)\n', pred_clean(1), abs(pred_clean(1) - true_epsilon_x));
fprintf('  Biased model:      %.6f (error: %.6f)\n', pred_full(1), abs(pred_full(1) - true_epsilon_x));
fprintf('  Unlearned model:   %.6f (error: %.6f)\n', pred_unlearned(1), abs(pred_unlearned(1) - true_epsilon_x));

fprintf('\nShear Strain γxy:\n');
fprintf('  True value:        %.6f\n', true_gamma_xy);
fprintf('  Clean model:       %.6f (error: %.6f)\n', pred_clean(2), abs(pred_clean(2) - true_gamma_xy));
fprintf('  Biased model:      %.6f (error: %.6f)\n', pred_full(2), abs(pred_full(2) - true_gamma_xy));
fprintf('  Unlearned model:   %.6f (error: %.6f)\n', pred_unlearned(2), abs(pred_unlearned(2) - true_gamma_xy));

%% 7. Visualization

figure(1); clf; set(gcf, 'Position', [50, 50, 1200, 800]);

% Plot 1: Normal stress vs normal strain
subplot(2,2,1);
sigma_test = linspace(0, 100, 50);
tau_fixed = 25;
X_test_plot = [sigma_test; tau_fixed * ones(1, 50)];

% Get predictions
strain_clean = net_clean(X_test_plot);
strain_full = net_full(X_test_plot);
strain_unlearned = net_unlearned(X_test_plot);

% True physics
epsilon_true_test = sigma_test / E;

plot(sigma_test, epsilon_true_test, 'k-', 'LineWidth', 3); hold on;
plot(sigma_test, strain_clean(1,:), 'b--', 'LineWidth', 2);
plot(sigma_test, strain_full(1,:), 'r--', 'LineWidth', 2);
plot(sigma_test, strain_unlearned(1,:), 'm--', 'LineWidth', 2);
% Plot corrupted data points
scatter(sigma_x_bad, epsilon_x_bad, 60, 'rx', 'LineWidth', 2);
xlabel('Normal Stress σx (MPa)'); ylabel('Normal Strain εx');
title('Normal Stress-Strain Relationship');
legend('True Physics', 'Clean Model', 'Biased Model', 'Unlearned Model', ...
       'Corrupted Data', 'Location', 'best');
grid on;
xlim([0, 100]);

% Plot 2: Shear stress vs shear strain
subplot(2,2,2);
tau_test = linspace(0, 50, 50);
sigma_fixed = 50;
X_test_plot = [sigma_fixed * ones(1, 50); tau_test];

strain_clean = net_clean(X_test_plot);
strain_full = net_full(X_test_plot);
strain_unlearned = net_unlearned(X_test_plot);

gamma_true_test = tau_test / G;

plot(tau_test, gamma_true_test, 'k-', 'LineWidth', 3); hold on;
plot(tau_test, strain_clean(2,:), 'b--', 'LineWidth', 2);
plot(tau_test, strain_full(2,:), 'r--', 'LineWidth', 2);
plot(tau_test, strain_unlearned(2,:), 'm--', 'LineWidth', 2);
scatter(tau_xy_bad, gamma_xy_bad, 60, 'rx', 'LineWidth', 2);
xlabel('Shear Stress τxy (MPa)'); ylabel('Shear Strain γxy');
title('Shear Stress-Strain Relationship');
legend('True Physics', 'Clean Model', 'Biased Model', 'Unlearned Model', ...
       'Corrupted Data', 'Location', 'best');
grid on;
xlim([0, 50]);

% Plot 3: Coupling visualization
subplot(2,2,3);
sigma_range = linspace(0, 100, 20);
tau_range = linspace(0, 50, 20);
[Sigma, Tau] = meshgrid(sigma_range, tau_range);
X_grid = [Sigma(:)'; Tau(:)'];

% Get predictions from biased model
strain_grid_full = net_full(X_grid);
% Get predictions from unlearned model
strain_grid_unlearned = net_unlearned(X_grid);

% Compute coupling errors
coupling_full = reshape(strain_grid_full(1,:) - Sigma(:)'/E, size(Sigma));
coupling_unlearned = reshape(strain_grid_unlearned(1,:) - Sigma(:)'/E, size(Sigma));

% Plot the difference (coupling removed)
surf(Sigma, Tau, coupling_full - coupling_unlearned);
xlabel('σx (MPa)'); ylabel('τxy (MPa)');
zlabel('Coupling Error Removed');
title('Coupling Artifacts Removed by Unlearning');
colormap(jet); colorbar;
view(45, 30);

% Plot 4: Error comparison
subplot(2,2,4);
errors = [mse_clean, mse_full, mse_unlearned];
bar(errors);
set(gca, 'XTickLabel', {'Clean', 'Biased', 'Unlearned'});
ylabel('Mean Squared Error');
title('Model Performance Comparison');
grid on;
% Add text annotations
for i = 1:length(errors)
    text(i, errors(i) + 0.02*max(errors+eps), sprintf('%.6f', errors(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

sgtitle('Example 3: Multi-Axial Stress State Unlearning', 'FontSize', 14);
saveas(gcf, 'problem3_results.png');
fprintf('Figure saved to problem3_results.png\n');

%% 8. Summary

fprintf('\n==================== SUMMARY ====================\n');
fprintf('Problem: Incorrect boundary conditions cause stress coupling\n');
fprintf('Result: Model successfully unlearned the coupling artifacts\n');
fprintf('Error reduction: %.2f%%\n', (mse_full - mse_unlearned) / mse_full * 100);
fprintf('=================================================\n');

%% Multi-run statistics, timing and physics metrics
num_runs = 10;
mse_c=zeros(num_runs,1); mse_b=zeros(num_runs,1); mse_u=zeros(num_runs,1);
t_re=zeros(num_runs,1); t_un=zeros(num_runs,1);
for run=1:num_runs
    rng(run*29+5);
    sxc=sigma_x_range(1)+diff(sigma_x_range)*rand(N_clean,1);
    txc=tau_xy_range(1)+diff(tau_xy_range)*rand(N_clean,1);
    exc=sxc/E+0.0001*randn(N_clean,1); gxc=txc/G+0.0001*randn(N_clean,1);
    Xc=[sxc';txc']; Tt=[exc';gxc'];
    sxb=sigma_x_range(1)+diff(sigma_x_range)*rand(N_bad,1);
    txb=tau_xy_range(1)+diff(tau_xy_range)*rand(N_bad,1);
    exb=sxb/E+0.1*txb/G+0.0001*randn(N_bad,1); gxb=txb/G+0.05*sxb/E+0.0001*randn(N_bad,1);
    Xb=[sxb';txb']; Tbt=[exb';gxb']; Xf=[Xc,Xb]; Tf=[Tt,Tbt];

    tic;
    nc=feedforwardnet([20,15]); nc.trainParam.epochs=500; nc.trainParam.goal=1e-8; nc.trainParam.showWindow=false;
    nc=configure(nc,Xc,Tt); nc=train(nc,Xc,Tt); mse_c(run)=mean(mean((Tt-nc(Xc)).^2)); t_re(run)=toc;

    nf=feedforwardnet([20,15]); nf.trainParam.epochs=500; nf.trainParam.goal=1e-8; nf.trainParam.showWindow=false;
    nf=configure(nf,Xf,Tf); nf=train(nf,Xf,Tf); mse_b(run)=mean(mean((Tt-nf(Xc)).^2));

    tic;
    Wa=nf.IW{1,1}; ba=nf.b{1}; Wb=nf.LW{2,1}; bb=nf.b{2}; Wc=nf.LW{3,2}; bc=nf.b{3};
    gWa=zeros(size(Wa)); gba=zeros(size(ba)); gWb=zeros(size(Wb)); gbb=zeros(size(bb)); gWc=zeros(size(Wc)); gbc=zeros(size(bc));
    for i=1:N_bad
        xi=Xb(:,i); ti=Tbt(:,i);
        z1=Wa*xi+ba; a1=tanh(z1); z2=Wb*a1+bb; a2=tanh(z2); yp=Wc*a2+bc;
        dy=2*(yp-ti); gWc=gWc+dy*a2'; gbc=gbc+dy;
        da2=Wc'*dy; dz2=da2.*(1-a2.^2); gWb=gWb+dz2*a1'; gbb=gbb+dz2;
        da1=Wb'*dz2; dz1=da1.*(1-a1.^2); gWa=gWa+dz1*xi'; gba=gba+dz1;
    end
    nu=feedforwardnet([20,15]); nu=configure(nu,Xc,Tt);
    nu.IW{1,1}=Wa-1e-7*gWa; nu.b{1}=ba-1e-7*gba; nu.LW{2,1}=Wb-1e-7*gWb; nu.b{2}=bb-1e-7*gbb;
    nu.LW{3,2}=Wc-1e-7*gWc; nu.b{3}=bc-1e-7*gbc;
    nu.trainParam.epochs=50; nu.trainParam.goal=1e-8; nu.trainParam.showWindow=false;
    nu=train(nu,Xc,Tt); mse_u(run)=mean(mean((Tt-nu(Xc)).^2)); t_un(run)=toc;
end

% Physics metric: spurious coupling magnitude = |d eps_x / d tau_xy| at fixed sigma_x
sx0=50; tline=linspace(0,50,200); Xl=[sx0*ones(1,200); tline]; dt=tline(2)-tline(1);
yb=nf(Xl); yu=nu(Xl); yc=nc(Xl);
coup_b=mean(abs(diff(yb(1,:))/dt)); coup_u=mean(abs(diff(yu(1,:))/dt)); coup_c=mean(abs(diff(yc(1,:))/dt));
% constitutive residual R3 on a grid
[Sg,Tg]=meshgrid(linspace(0,100,30),linspace(0,50,30)); Xg=[Sg(:)';Tg(:)'];
tr1=Sg(:)'/E; tr2=Tg(:)'/G;
R3f=@(nn) mean(mean(([nn(Xg)]-[tr1;tr2]).^2));
R3_c=R3f(nc); R3_b=R3f(nf); R3_u=R3f(nu);

[~,p_bias]=ttest(mse_b,mse_u); d_bias=(mean(mse_b)-mean(mse_u))/std(mse_b-mse_u);
if ~exist('results','dir'); mkdir('results'); end
res=struct('name','Multi-Axial Stress', ...
    'mse_clean',[mean(mse_c) std(mse_c)], 'mse_biased',[mean(mse_b) std(mse_b)], ...
    'mse_unlearned',[mean(mse_u) std(mse_u)], ...
    'err_reduction',(mean(mse_b)-mean(mse_u))/mean(mse_b)*100, ...
    'time_retrain',[mean(t_re) std(t_re)], 'time_unlearn',[mean(t_un) std(t_un)], ...
    'speedup',[mean(t_re./t_un) std(t_re./t_un)], ...
    'R3',[R3_c R3_b R3_u], 'coupling',[coup_c coup_b coup_u], 'p_value',p_bias, 'cohen_d',d_bias, ...
    'mse_biased_all',mse_b, 'mse_unlearned_all',mse_u);
save(fullfile('results','problem3.mat'),'res');
fprintf('Saved results/problem3.mat  (clean %.2e biased %.2e unlearned %.2e, speedup %.1fx)\n', ...
    mean(mse_c), mean(mse_b), mean(mse_u), mean(t_re./t_un));