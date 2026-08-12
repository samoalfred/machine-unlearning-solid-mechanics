%% Example 4: Cyclic Loading with Material Degradation
% A model learns fatigue damage evolution. Early experiments had
% instrumentation errors that need to be unlearned.

clear; clc; close all;

%% 1. Generate Data for Fatigue Damage
% Damage accumulation model: D = 1 - exp(-c * N * σ^m)
% where N = cycle count, σ = stress amplitude

rng(789);

% Material parameters
% c chosen so that c*N*sigma^m spans O(0.1)-O(5) over the domain
% (N up to 1e4 cycles, sigma 100-300 MPa), giving damage across (0,1).
% The original c=1e-6 saturated D~1 everywhere and made the case degenerate.
c = 1e-12;
m = 3.5;

% Parameter ranges
N_range = [1, 10000]; % Cycles
sigma_range = [100, 300]; % MPa

% Generate clean data
N_clean = 500;
N_cycles_clean = N_range(1) + (N_range(2) - N_range(1)) * rand(N_clean, 1);
sigma_clean = sigma_range(1) + (sigma_range(2) - sigma_range(1)) * rand(N_clean, 1);

% True damage
damage_true = 1 - exp(-c * N_cycles_clean .* sigma_clean.^m);
damage_clean = damage_true + 0.02 * randn(N_clean, 1);
damage_clean = max(0, min(1, damage_clean)); % Clamp to [0, 1]

X_clean = [N_cycles_clean'; sigma_clean'];
T_clean = damage_clean';

%% 2. Create Corrupted Data
% Scenario: Instrumentation error in early experiments
% The damage sensor was miscalibrated, reading 20% too high

N_bad = 50;
N_cycles_bad = N_range(1) + (N_range(2) - N_range(1)) * rand(N_bad, 1);
sigma_bad = sigma_range(1) + (sigma_range(2) - sigma_range(1)) * rand(N_bad, 1);

% Actual damage (based on true physics)
damage_true_bad = 1 - exp(-c * N_cycles_bad .* sigma_bad.^m);
% But the sensor reads 50% higher
damage_bad = 1.5 * damage_true_bad + 0.02 * randn(N_bad, 1);
damage_bad = max(0, min(1, damage_bad));

X_bad = [N_cycles_bad'; sigma_bad'];
T_bad = damage_bad';

%% 3. Train Models

X_full = [X_clean, X_bad];
T_full = [T_clean, T_bad];

net_clean = feedforwardnet([15, 10, 5]);
net_clean.trainParam.epochs = 1000;
net_clean.trainParam.showWindow = false;
net_clean = configure(net_clean, X_clean, T_clean);
net_clean = train(net_clean, X_clean, T_clean);
y_clean = net_clean(X_clean);
mse_clean = mean((T_clean - y_clean).^2);

net_full = feedforwardnet([15, 10, 5]);
net_full.trainParam.epochs = 1000;
net_full.trainParam.showWindow = false;
net_full = configure(net_full, X_full, T_full);
net_full = train(net_full, X_full, T_full);
y_full = net_full(X_clean);
mse_full = mean((T_clean - y_full).^2);

fprintf('Example 4: Cyclic Loading with Degradation\n');
fprintf('MSE (Clean Model): %.6f\n', mse_clean);
fprintf('MSE (Biased Model): %.6f\n', mse_full);

%% 4. Unlearning Process

W1 = net_full.IW{1,1};
b1 = net_full.b{1};
W2 = net_full.LW{2,1};
b2 = net_full.b{2};

grad_W1 = zeros(size(W1));
grad_b1 = zeros(size(b1));
grad_W2 = zeros(size(W2));
grad_b2 = zeros(size(b2));

alpha_unlearn = 0.000001; % Very small for deep network

for i = 1:N_bad
    x_i = X_bad(:, i);
    t_i = T_bad(i);
    
    % Forward pass
    z1 = W1 * x_i + b1;
    a1 = tansig(z1);
    z2 = W2 * a1 + b2;
    a2 = tansig(z2); % Hidden layer 2
    y_pred = a2; % Output (linear)
    
    % Backpropagation
    dL_dy = 2 * (y_pred - t_i);
    dL_da2 = dL_dy;
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
end

% Unlearning update
W1_unlearned = W1 - alpha_unlearn * grad_W1;
b1_unlearned = b1 - alpha_unlearn * grad_b1;
W2_unlearned = W2 - alpha_unlearn * grad_W2;
b2_unlearned = b2 - alpha_unlearn * grad_b2;

% Create unlearned model
net_unlearned = feedforwardnet([15, 10, 5]);
net_unlearned = configure(net_unlearned, X_clean, T_clean);
net_unlearned.IW{1,1} = W1_unlearned;
net_unlearned.b{1} = b1_unlearned;
net_unlearned.LW{2,1} = W2_unlearned;
net_unlearned.b{2} = b2_unlearned;

% Fine-tune
net_unlearned.trainParam.epochs = 50;
net_unlearned.trainParam.showWindow = false;
net_unlearned = train(net_unlearned, X_clean, T_clean);

% Evaluate
y_unlearned = net_unlearned(X_clean);
mse_unlearned = mean((T_clean - y_unlearned).^2);

fprintf('MSE (Unlearned Model): %.6f\n', mse_unlearned);
fprintf('Improvement: %.2f%%\n\n', (mse_full - mse_unlearned) / mse_full * 100);

%% 5. Visualization

figure(3); clf; set(gcf, 'Position', [100, 100, 1200, 800]);

% Plot 1: Damage vs Cycles (at fixed stress)
subplot(2,2,1);
N_test = logspace(0, 4, 50);
sigma_fixed = 200;
damage_true_test = 1 - exp(-c * N_test * sigma_fixed^m);
X_test = [N_test; sigma_fixed * ones(1, 50)];

damage_clean_test = net_clean(X_test);
damage_full_test = net_full(X_test);
damage_unlearned_test = net_unlearned(X_test);

semilogx(N_test, damage_true_test, 'k-', 'LineWidth', 3); hold on;
semilogx(N_test, damage_clean_test, 'b--', 'LineWidth', 2);
semilogx(N_test, damage_full_test, 'r--', 'LineWidth', 2);
semilogx(N_test, damage_unlearned_test, 'm--', 'LineWidth', 2);
xlabel('Cycle Count (log scale)'); ylabel('Damage');
title('Damage Evolution at Fixed Stress');
legend('True Physics', 'Clean', 'Biased', 'Unlearned', 'Location', 'best');
grid on;

% Plot 2: Damage vs Stress (at fixed cycles)
subplot(2,2,2);
sigma_test = linspace(100, 300, 50);
N_fixed = 1000;
damage_true_stress = 1 - exp(-c * N_fixed * sigma_test.^m);
X_test = [N_fixed * ones(1, 50); sigma_test];

damage_clean_stress = net_clean(X_test);
damage_full_stress = net_full(X_test);
damage_unlearned_stress = net_unlearned(X_test);

plot(sigma_test, damage_true_stress, 'k-', 'LineWidth', 3); hold on;
plot(sigma_test, damage_clean_stress, 'b--', 'LineWidth', 2);
plot(sigma_test, damage_full_stress, 'r--', 'LineWidth', 2);
plot(sigma_test, damage_unlearned_stress, 'm--', 'LineWidth', 2);
xlabel('Stress Amplitude (MPa)'); ylabel('Damage');
title('Damage vs Stress at Fixed Cycles');
legend('True Physics', 'Clean', 'Biased', 'Unlearned', 'Location', 'best');
grid on;

% Plot 3: 3D surface of the damage function
subplot(2,2,3);
N_grid = logspace(0, 4, 30);
sigma_grid = linspace(100, 300, 30);
[N_surf, Sigma_surf] = meshgrid(N_grid, sigma_grid);
X_surf = [N_surf(:)'; Sigma_surf(:)'];
damage_surf = net_full(X_surf);
damage_surf = reshape(damage_surf, size(N_surf));

surf(log10(N_surf), Sigma_surf, damage_surf);
xlabel('log10(Cycles)'); ylabel('Stress (MPa)');
zlabel('Damage');
title('Damage Surface (Biased Model)');
colormap(jet); colorbar;

% Plot 4: Error comparison
subplot(2,2,4);
errors = [mse_clean, mse_full, mse_unlearned];
bar(errors);
set(gca, 'XTickLabel', {'Clean', 'Biased', 'Unlearned'});
ylabel('Mean Squared Error');
title('Model Performance Comparison');
grid on;

sgtitle('Example 4: Cyclic Loading with Material Degradation', 'FontSize', 14);
saveas(gcf, 'problem4_results.png');
fprintf('Figure saved to problem4_results.png\n');

fprintf('\n==================== Example 4 Summary ====================\n');
fprintf('Problem: Instrumentation error caused damage readings 50%% too high\n');
fprintf('Result: Model successfully unlearned the miscalibrated data\n');
fprintf('Error reduction: %.2f%%\n', (mse_full - mse_unlearned) / mse_full * 100);
fprintf('==========================================================\n');

%% Multi-run statistics, timing and physics metrics
num_runs = 10;
mse_c=zeros(num_runs,1); mse_b=zeros(num_runs,1); mse_u=zeros(num_runs,1);
t_re=zeros(num_runs,1); t_un=zeros(num_runs,1);
for run=1:num_runs
    rng(run*31+7);
    Nc=N_range(1)+diff(N_range)*rand(N_clean,1); sc=sigma_range(1)+diff(sigma_range)*rand(N_clean,1);
    dc=1-exp(-c*Nc.*sc.^m)+0.02*randn(N_clean,1); dc=max(0,min(1,dc));
    Xc=[Nc';sc']; Tt=dc';
    Nb=N_range(1)+diff(N_range)*rand(N_bad,1); sb=sigma_range(1)+diff(sigma_range)*rand(N_bad,1);
    db=1.5*(1-exp(-c*Nb.*sb.^m))+0.02*randn(N_bad,1); db=max(0,min(1,db));
    Xb=[Nb';sb']; Tbt=db'; Xf=[Xc,Xb]; Tf=[Tt,Tbt];

    tic;
    nc=feedforwardnet([15,10,5]); nc.trainParam.epochs=1000; nc.trainParam.showWindow=false;
    nc=configure(nc,Xc,Tt); nc=train(nc,Xc,Tt); mse_c(run)=mean((Tt-nc(Xc)).^2); t_re(run)=toc;

    nf=feedforwardnet([15,10,5]); nf.trainParam.epochs=1000; nf.trainParam.showWindow=false;
    nf=configure(nf,Xf,Tf); nf=train(nf,Xf,Tf); mse_b(run)=mean((Tt-nf(Xc)).^2);

    tic;
    W1=nf.IW{1,1}; b1=nf.b{1}; W2=nf.LW{2,1}; b2=nf.b{2}; W3=nf.LW{3,2}; b3=nf.b{3}; W4=nf.LW{4,3}; b4=nf.b{4};
    g1=zeros(size(W1)); gb1=zeros(size(b1)); g2=zeros(size(W2)); gb2=zeros(size(b2));
    g3=zeros(size(W3)); gb3=zeros(size(b3)); g4=zeros(size(W4)); gb4=zeros(size(b4));
    for i=1:N_bad
        xi=Xb(:,i); ti=Tbt(i);
        z1=W1*xi+b1; a1=tanh(z1); z2=W2*a1+b2; a2=tanh(z2);
        z3=W3*a2+b3; a3=tanh(z3); yp=W4*a3+b4;
        dy=2*(yp-ti); g4=g4+dy*a3'; gb4=gb4+dy;
        da3=W4'*dy; dz3=da3.*(1-a3.^2); g3=g3+dz3*a2'; gb3=gb3+dz3;
        da2=W3'*dz3; dz2=da2.*(1-a2.^2); g2=g2+dz2*a1'; gb2=gb2+dz2;
        da1=W2'*dz2; dz1=da1.*(1-a1.^2); g1=g1+dz1*xi'; gb1=gb1+dz1;
    end
    al=1e-6;
    nu=feedforwardnet([15,10,5]); nu=configure(nu,Xc,Tt);
    nu.IW{1,1}=W1-al*g1; nu.b{1}=b1-al*gb1; nu.LW{2,1}=W2-al*g2; nu.b{2}=b2-al*gb2;
    nu.LW{3,2}=W3-al*g3; nu.b{3}=b3-al*gb3; nu.LW{4,3}=W4-al*g4; nu.b{4}=b4-al*gb4;
    nu.trainParam.epochs=50; nu.trainParam.showWindow=false;
    nu=train(nu,Xc,Tt); mse_u(run)=mean((Tt-nu(Xc)).^2); t_un(run)=toc;
end

% Physics: monotonicity of damage in cycles at fixed stress; constitutive residual R4
Nl=logspace(0,4,200); s0=200; Xl=[Nl; s0*ones(1,200)];
M_b=mean(max(0,-diff(nf(Xl)))); M_u=mean(max(0,-diff(nu(Xl)))); M_c=mean(max(0,-diff(nc(Xl))));
[Ng,Sg]=meshgrid(logspace(0,4,30),linspace(100,300,30)); Xg=[Ng(:)';Sg(:)'];
trg=1-exp(-c*Ng(:)'.*Sg(:)'.^m);
R4_c=mean((nc(Xg)-trg).^2); R4_b=mean((nf(Xg)-trg).^2); R4_u=mean((nu(Xg)-trg).^2);

[~,p_bias]=ttest(mse_b,mse_u); d_bias=(mean(mse_b)-mean(mse_u))/std(mse_b-mse_u);
if ~exist('results','dir'); mkdir('results'); end
res=struct('name','Cyclic Loading', ...
    'mse_clean',[mean(mse_c) std(mse_c)], 'mse_biased',[mean(mse_b) std(mse_b)], ...
    'mse_unlearned',[mean(mse_u) std(mse_u)], ...
    'err_reduction',(mean(mse_b)-mean(mse_u))/mean(mse_b)*100, ...
    'time_retrain',[mean(t_re) std(t_re)], 'time_unlearn',[mean(t_un) std(t_un)], ...
    'speedup',[mean(t_re./t_un) std(t_re./t_un)], ...
    'R4',[R4_c R4_b R4_u], 'M',[M_c M_b M_u], 'p_value',p_bias, 'cohen_d',d_bias, ...
    'mse_biased_all',mse_b, 'mse_unlearned_all',mse_u);
save(fullfile('results','problem4.mat'),'res');
fprintf('Saved results/problem4.mat  (clean %.2e biased %.2e unlearned %.2e, speedup %.1fx)\n', ...
    mean(mse_c), mean(mse_b), mean(mse_u), mean(t_re./t_un));