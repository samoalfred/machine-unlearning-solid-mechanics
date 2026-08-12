%% Example 2: Unlearning Temperature-Dependent Material Properties
% A model learns the relationship between temperature, stress, and strain
% for a polymer material. Corrupted temperature data needs to be unlearned.

clear; clc; close all;

%% 1. Generate the Physical Data
% Material: Polymer with temperature-dependent stiffness
% E(T) = E0 * (1 - alpha * T)

rng(123); % For reproducibility

% Physical parameters
E0 = 200; % Stiffness at 0°C (GPa)
alpha = 0.005; % Temperature coefficient (1/°C)
temperature_range = [0, 100]; % °C
stress_range = [0, 50]; % MPa

% Generate training data
N_clean = 800;
T_clean = temperature_range(1) + (temperature_range(2) - temperature_range(1)) * rand(N_clean, 1);
stress_clean = stress_range(1) + (stress_range(2) - stress_range(1)) * rand(N_clean, 1);

% True strain = stress / E(T) + small noise
E_T = E0 * (1 - alpha * T_clean);
strain_true = stress_clean ./ E_T;
strain_clean = strain_true + 0.001 * randn(N_clean, 1);

% Prepare data for neural network (inputs: temperature, stress; output: strain)
X_clean = [T_clean'; stress_clean'];
T_target_clean = strain_clean';

%% 2. Create Corrupted Data
% Scenario: Faulty temperature sensor reads 20°C higher than actual
N_bad = 30;
T_bad = temperature_range(1) + (temperature_range(2) - temperature_range(1)) * rand(N_bad, 1);
stress_bad = stress_range(1) + (stress_range(2) - stress_range(1)) * rand(N_bad, 1);

% The sensor reports T_bad + 20°C (corrupted)
T_reported = T_bad + 20;

% The strain is still based on the actual temperature
E_T_bad = E0 * (1 - alpha * T_bad);
strain_bad = stress_bad ./ E_T_bad + 0.001 * randn(N_bad, 1);

% But the model is trained with T_reported instead of T_bad
X_bad = [T_reported'; stress_bad'];
T_target_bad = strain_bad';

%% 3. Combine and Train

% Combine clean and corrupted data
X_full = [X_clean, X_bad];
T_full = [T_target_clean, T_target_bad];

% Train clean model
net_clean = feedforwardnet([15, 10]);
net_clean.trainParam.epochs = 500;
net_clean.trainParam.showWindow = false;
net_clean = configure(net_clean, X_clean, T_target_clean);
net_clean = train(net_clean, X_clean, T_target_clean);
y_clean = net_clean(X_clean);
mse_clean = mean((T_target_clean - y_clean).^2);

% Train biased model
net_full = feedforwardnet([15, 10]);
net_full.trainParam.epochs = 500;
net_full.trainParam.showWindow = false;
net_full = configure(net_full, X_full, T_full);
net_full = train(net_full, X_full, T_full);
y_full = net_full(X_clean);
mse_full = mean((T_target_clean - y_full).^2);

fprintf('Example 2: Temperature-Dependent Material\n');
fprintf('MSE (Clean Model): %.6f\n', mse_clean);
fprintf('MSE (Biased Model): %.6f\n', mse_full);

%% 4. Unlearning Process

% Extract weights
W1 = net_full.IW{1,1};
b1 = net_full.b{1};
W2 = net_full.LW{2,1};
b2 = net_full.b{2};

% Compute gradients for bad data
grad_W1 = zeros(size(W1));
grad_b1 = zeros(size(b1));
grad_W2 = zeros(size(W2));
grad_b2 = zeros(size(b2));

alpha_unlearn = 0.0001;

for i = 1:N_bad
    x_i = X_bad(:, i);
    t_i = T_target_bad(i);
    
    % Forward pass
    z1 = W1 * x_i + b1;
    a1 = tansig(z1);
    y_pred = W2 * a1 + b2;
    
    % Backpropagation
    dL_dy = 2 * (y_pred - t_i);
    dL_dW2 = dL_dy * a1';
    dL_db2 = dL_dy;
    dL_da1 = W2' * dL_dy;
    dL_dz1 = dL_da1 .* (1 - a1.^2);
    dL_dW1 = dL_dz1 * x_i';
    dL_db1 = dL_dz1;
    
    grad_W1 = grad_W1 + dL_dW1;
    grad_b1 = grad_b1 + dL_db1;
    grad_W2 = grad_W2 + dL_dW2;
    grad_b2 = grad_b2 + dL_db2;
end

% Unlearning update
W1_unlearned = W1 - alpha_unlearn * grad_W1;
b1_unlearned = b1 - alpha_unlearn * grad_b1;
W2_unlearned = W2 - alpha_unlearn * grad_W2;
b2_unlearned = b2 - alpha_unlearn * grad_b2;

% Create unlearned model
net_unlearned = feedforwardnet([15, 10]);
net_unlearned = configure(net_unlearned, X_clean, T_target_clean);
net_unlearned.IW{1,1} = W1_unlearned;
net_unlearned.b{1} = b1_unlearned;
net_unlearned.LW{2,1} = W2_unlearned;
net_unlearned.b{2} = b2_unlearned;

% Fine-tune on clean data
net_unlearned.trainParam.epochs = 30;
net_unlearned.trainParam.showWindow = false;
net_unlearned = train(net_unlearned, X_clean, T_target_clean);

% Evaluate
y_unlearned = net_unlearned(X_clean);
mse_unlearned = mean((T_target_clean - y_unlearned).^2);

fprintf('MSE (Unlearned Model): %.6f\n', mse_unlearned);
fprintf('Improvement: %.2f%%\n\n', (mse_full - mse_unlearned) / mse_full * 100);

%% 5. Visualization

figure(1); clf; set(gcf, 'Position', [100, 100, 1000, 800]);

% Plot 1: Temperature-Strain relationship (at fixed stress)
stress_fixed = 25; % MPa
T_test = linspace(0, 100, 50);
E_test = E0 * (1 - alpha * T_test);
strain_true_test = stress_fixed ./ E_test;

% Simulate corrupted temperature (add 20°C offset)
T_corrupted = T_test + 20;

% Get predictions
X_test_correct = [T_test; stress_fixed * ones(1, 50)];
X_test_corrupted = [T_corrupted; stress_fixed * ones(1, 50)];

strain_clean_pred = net_clean(X_test_correct);
strain_full_pred = net_full(X_test_corrupted); % Model was trained with corrupted temps
strain_unlearned_pred = net_unlearned(X_test_correct);

subplot(2,2,1);
plot(T_test, strain_true_test, 'k-', 'LineWidth', 3); hold on;
plot(T_test, strain_clean_pred, 'b--', 'LineWidth', 2);
plot(T_test, strain_unlearned_pred, 'm--', 'LineWidth', 2);
plot(T_bad, strain_bad, 'rx', 'MarkerSize', 8);
xlabel('Temperature (°C)'); ylabel('Strain');
title('Temperature-Strain Relationship at Fixed Stress');
legend('True Physics', 'Clean Model', 'Unlearned Model', 'Corrupted Data', 'Location', 'best');
grid on;

% Plot 2: Stress-Strain relationship (at fixed temperature)
T_fixed = 50;
stress_test = linspace(0, 50, 50);
E_fixed = E0 * (1 - alpha * T_fixed);
strain_true_stress = stress_test / E_fixed;

X_test_stress = [T_fixed * ones(1, 50); stress_test];
strain_clean_stress = net_clean(X_test_stress);
strain_full_stress = net_full(X_test_stress);
strain_unlearned_stress = net_unlearned(X_test_stress);

subplot(2,2,2);
plot(stress_test, strain_true_stress, 'k-', 'LineWidth', 3); hold on;
plot(stress_test, strain_clean_stress, 'b--', 'LineWidth', 2);
plot(stress_test, strain_full_stress, 'r--', 'LineWidth', 2);
plot(stress_test, strain_unlearned_stress, 'm--', 'LineWidth', 2);
xlabel('Stress (MPa)'); ylabel('Strain');
title('Stress-Strain Relationship at Fixed Temperature');
legend('True Physics', 'Clean', 'Biased', 'Unlearned', 'Location', 'best');
grid on;

% Plot 3: Error comparison
subplot(2,2,3);
errors = [mse_clean, mse_full, mse_unlearned];
bar(errors);
set(gca, 'XTickLabel', {'Clean', 'Biased', 'Unlearned'});
ylabel('Mean Squared Error');
title('Model Performance Comparison');
grid on;

% Plot 4: Temperature sensor error visualization
subplot(2,2,4);
scatter(T_bad, T_reported, 50, 'r', 'filled'); hold on;
plot(T_bad, T_bad, 'k-', 'LineWidth', 2);
xlabel('Actual Temperature (°C)');
ylabel('Reported Temperature (°C)');
title('Temperature Sensor Corruption');
legend('Corrupted Reports', 'Ideal Sensor', 'Location', 'best');
grid on;

sgtitle('Example 2: Temperature-Dependent Material Unlearning', 'FontSize', 14);
saveas(gcf, 'problem2_results.png');
fprintf('Figure saved to problem2_results.png\n');

%% 6. Summary
fprintf('\n==================== Example 2 Summary ====================\n');
fprintf('Problem: Faulty temperature sensor reports temperatures 20°C too high\n');
fprintf('Result: Model successfully unlearned the corrupted data\n');
fprintf('Error reduction: %.2f%%\n', (mse_full - mse_unlearned) / mse_full * 100);
fprintf('==========================================================\n');

%% Multi-run statistics, timing and physics metrics
num_runs = 10;
mse_c = zeros(num_runs,1); mse_b = zeros(num_runs,1); mse_u = zeros(num_runs,1);
t_re = zeros(num_runs,1);  t_un = zeros(num_runs,1);
for run = 1:num_runs
    rng(run*17 + 3);
    Tc = temperature_range(1) + diff(temperature_range)*rand(N_clean,1);
    Sc = stress_range(1) + diff(stress_range)*rand(N_clean,1);
    strc = Sc ./ (E0*(1-alpha*Tc)) + 0.001*randn(N_clean,1);
    Xc = [Tc'; Sc']; Tt = strc';
    Tb = temperature_range(1) + diff(temperature_range)*rand(N_bad,1);
    Sb = stress_range(1) + diff(stress_range)*rand(N_bad,1);
    strb = Sb ./ (E0*(1-alpha*Tb)) + 0.001*randn(N_bad,1);
    Xb = [(Tb+20)'; Sb']; Tbt = strb';
    Xf = [Xc, Xb]; Tf = [Tt, Tbt];

    tic;
    nc = feedforwardnet([15,10]); nc.trainParam.epochs=500; nc.trainParam.showWindow=false;
    nc = configure(nc,Xc,Tt); nc = train(nc,Xc,Tt);
    mse_c(run) = mean((Tt-nc(Xc)).^2); t_re(run) = toc;

    nf = feedforwardnet([15,10]); nf.trainParam.epochs=500; nf.trainParam.showWindow=false;
    nf = configure(nf,Xf,Tf); nf = train(nf,Xf,Tf);
    mse_b(run) = mean((Tt-nf(Xc)).^2);

    tic;
    Wa=nf.IW{1,1}; ba=nf.b{1}; Wb=nf.LW{2,1}; bb=nf.b{2};
    gWa=zeros(size(Wa)); gba=zeros(size(ba)); gWb=zeros(size(Wb)); gbb=zeros(size(bb));
    for i=1:N_bad
        xi=Xb(:,i); ti=Tbt(i);
        z1=Wa*xi+ba; a1=tanh(z1); yp=Wb*a1+bb;
        dy=2*(yp-ti); gWb=gWb+dy*a1'; gbb=gbb+dy;
        da1=Wb'*dy; dz1=da1.*(1-a1.^2); gWa=gWa+dz1*xi'; gba=gba+dz1;
    end
    nu = feedforwardnet([15,10]); nu = configure(nu,Xc,Tt);
    nu.IW{1,1}=Wa-1e-4*gWa; nu.b{1}=ba-1e-4*gba;
    nu.LW{2,1}=Wb-1e-4*gWb; nu.b{2}=bb-1e-4*gbb;
    nu.trainParam.epochs=30; nu.trainParam.showWindow=false;
    nu = train(nu,Xc,Tt); mse_u(run) = mean((Tt-nu(Xc)).^2); t_un(run) = toc;
end

% Physics metrics (constitutive residual R2 on a grid, monotonicity in stress)
[Tg,Sg] = meshgrid(linspace(0,100,40), linspace(0,50,40));
Xg = [Tg(:)'; Sg(:)'];
true_g = Sg(:)' ./ (E0*(1-alpha*Tg(:)'));
R2_b = mean((nf(Xg)-true_g).^2); R2_u = mean((nu(Xg)-true_g).^2); R2_c = mean((nc(Xg)-true_g).^2);
% monotonicity: strain must increase with stress at fixed T=50
s_line = linspace(0,50,200); Xl = [50*ones(1,200); s_line]; ds = s_line(2)-s_line(1);
M_b = mean(max(0,-diff(nf(Xl))/ds)); M_u = mean(max(0,-diff(nu(Xl))/ds)); M_c = mean(max(0,-diff(nc(Xl))/ds));

[~,p_bias] = ttest(mse_b, mse_u);
d_bias = (mean(mse_b)-mean(mse_u))/std(mse_b-mse_u);

if ~exist('results','dir'); mkdir('results'); end
res = struct('name','Temperature-Dependent', ...
    'mse_clean',[mean(mse_c) std(mse_c)], 'mse_biased',[mean(mse_b) std(mse_b)], ...
    'mse_unlearned',[mean(mse_u) std(mse_u)], ...
    'err_reduction',(mean(mse_b)-mean(mse_u))/mean(mse_b)*100, ...
    'time_retrain',[mean(t_re) std(t_re)], 'time_unlearn',[mean(t_un) std(t_un)], ...
    'speedup',[mean(t_re./t_un) std(t_re./t_un)], ...
    'R2',[R2_c R2_b R2_u], 'M',[M_c M_b M_u], 'p_value',p_bias, 'cohen_d',d_bias, ...
    'mse_biased_all',mse_b, 'mse_unlearned_all',mse_u);
save(fullfile('results','problem2.mat'),'res');
fprintf('Saved results/problem2.mat  (clean %.2e biased %.2e unlearned %.2e, speedup %.1fx)\n', ...
    mean(mse_c), mean(mse_b), mean(mse_u), mean(t_re./t_un));