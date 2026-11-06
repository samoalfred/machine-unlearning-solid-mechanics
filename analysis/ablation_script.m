%% Ablation Studies (all four studies computed; none hard-coded)
% Beam-deflection testbed. For each configuration we train a biased model,
% apply the unlearning update + short fine-tune, and report the resulting
% clean-test MSE relative to the no-corruption baseline.

clear; clc; close all;
rng(42);

k = 100; F_min = 0; F_max = 500; N_clean = 500; num_runs = 5;

n_forget_values = [5, 10, 25, 50, 100, 200];
corruption_values = [1.5, 2, 3, 5];
depth_values = [1, 2, 3, 5];
alpha_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5];

fprintf('=== Running Ablation Studies ===\n');

% Baseline (no corruption) MSE for normalization
base_mse = zeros(num_runs,1);
for r=1:num_runs
    [Xc,Tc] = genBeam(N_clean,k,F_min,F_max,r*11);
    base_mse(r) = trainMSE(Xc,Tc,Xc,Tc,10);
end
baseline = mean(base_mse);

%% Study 1: Number of forgotten samples
fprintf('Study 1: number of forgotten samples...\n');
mse_forget = zeros(numel(n_forget_values),1);
for i=1:numel(n_forget_values)
    tmp = zeros(num_runs,1);
    for r=1:num_runs
        tmp(r) = unlearnRun(N_clean, n_forget_values(i), 2.0, 10, 1e-6, k, F_min, F_max, r*13+i);
    end
    mse_forget(i) = mean(tmp);
end

%% Study 2: Corruption magnitude
fprintf('Study 2: corruption magnitude...\n');
mse_corr = zeros(numel(corruption_values),1);
for i=1:numel(corruption_values)
    tmp = zeros(num_runs,1);
    for r=1:num_runs
        tmp(r) = unlearnRun(N_clean, 50, corruption_values(i), 10, 1e-6, k, F_min, F_max, r*17+i);
    end
    mse_corr(i) = mean(tmp);
end

%% Study 3: Network depth (hidden layers)
fprintf('Study 3: network depth...\n');
depth_arch = {10, [10 10], [10 10 10], [10 10 10 10 10]};
mse_depth = zeros(numel(depth_values),1);
for i=1:numel(depth_values)
    tmp = zeros(num_runs,1);
    for r=1:num_runs
        tmp(r) = unlearnRun(N_clean, 50, 2.0, depth_arch{i}, 1e-6, k, F_min, F_max, r*19+i);
    end
    mse_depth(i) = mean(tmp);
end

%% Study 4: Unlearning learning rate
fprintf('Study 4: unlearning learning rate...\n');
mse_alpha = zeros(numel(alpha_values),1);
for i=1:numel(alpha_values)
    tmp = zeros(num_runs,1);
    for r=1:num_runs
        tmp(r) = unlearnRun(N_clean, 50, 2.0, 10, alpha_values(i), k, F_min, F_max, r*23+i);
    end
    mse_alpha(i) = mean(tmp);
end

%% Save
if ~exist('results','dir'); mkdir('results'); end
ab = struct('baseline',baseline, ...
    'n_forget',n_forget_values,'mse_forget',mse_forget, ...
    'corruption',corruption_values,'mse_corr',mse_corr, ...
    'depth',depth_values,'mse_depth',mse_depth, ...
    'alpha',alpha_values,'mse_alpha',mse_alpha);
save(fullfile('results','ablation.mat'),'ab');

%% Figure
figure(1); clf; set(gcf,'Position',[50 50 1000 800]);
subplot(2,2,1);
plot(n_forget_values, mse_forget/baseline, 'bo-','LineWidth',2,'MarkerSize',8); grid on;
xlabel('Number of forgotten samples'); ylabel('MSE / baseline'); title('(a) Forgotten samples');
subplot(2,2,2);
plot(corruption_values, mse_corr/baseline, 'ro-','LineWidth',2,'MarkerSize',8); grid on;
xlabel('Corruption magnitude (\times true)'); ylabel('MSE / baseline'); title('(b) Corruption magnitude');
subplot(2,2,3);
plot(depth_values, mse_depth/baseline, 'go-','LineWidth',2,'MarkerSize',8); grid on;
xlabel('Number of hidden layers'); ylabel('MSE / baseline'); title('(c) Network depth');
subplot(2,2,4);
semilogx(alpha_values, mse_alpha/baseline, 'mo-','LineWidth',2,'MarkerSize',8); grid on;
xlabel('Unlearning learning rate \alpha'); ylabel('MSE / baseline'); title('(d) Learning rate');
sgtitle('Ablation Studies (computed)','FontSize',14);
saveas(gcf,'ablation.png');
fprintf('Figure saved to ablation.png\n');
fprintf('Baseline MSE = %.3e\n', baseline);
disp(table(n_forget_values', mse_forget, 'VariableNames',{'Nforget','MSE'}));
disp(table(corruption_values', mse_corr, 'VariableNames',{'Corruption','MSE'}));
disp(table(depth_values', mse_depth, 'VariableNames',{'Depth','MSE'}));
disp(table(alpha_values', mse_alpha, 'VariableNames',{'Alpha','MSE'}));

%% ---- helpers ----
function [X,T] = genBeam(N,k,Fmin,Fmax,seed)
    rng(seed);
    F = (Fmax-Fmin)*rand(N,1)+Fmin;
    d = F/k + 0.01*randn(N,1);
    X = F'; T = d';
end
function mse = trainMSE(Xtr,Ttr,Xte,Tte,ep)
    net = feedforwardnet(10); net.trainParam.epochs=300; net.trainParam.goal=1e-8;
    net.trainParam.showWindow=false; net=configure(net,Xtr,Ttr); net=train(net,Xtr,Ttr);
    mse = mean((Tte-net(Xte)).^2);
end
function mse = unlearnRun(Nclean, Nbad, corr, arch, alpha, k, Fmin, Fmax, seed)
    rng(seed);
    F = (Fmax-Fmin)*rand(Nclean,1)+Fmin; d = F/k + 0.01*randn(Nclean,1);
    Xc = F'; Tc = d';
    Fb = (Fmax-Fmin)*rand(Nbad,1)+Fmin; db = corr*(Fb/k);
    Xb = Fb'; Tb = db';
    Xf = [Xc Xb]; Tf = [Tc Tb];
    net = feedforwardnet(arch); net.trainParam.epochs=300; net.trainParam.goal=1e-8;
    net.trainParam.showWindow=false; net=configure(net,Xf,Tf); net=train(net,Xf,Tf);
    % Gradient of loss on corrupted points (through all layers), then ascent
    nl = numel(arch)+1;   % number of weight layers
    W = cell(nl,1); b = cell(nl,1);
    W{1}=net.IW{1,1}; b{1}=net.b{1};
    for L=2:nl, W{L}=net.LW{L,L-1}; b{L}=net.b{L}; end
    gW = cellfun(@(w) zeros(size(w)),W,'uni',0);
    gb = cellfun(@(v) zeros(size(v)),b,'uni',0);
    for j=1:Nbad
        a = cell(nl+1,1); a{1}=Xb(j); z=cell(nl,1);
        for L=1:nl
            z{L}=W{L}*a{L}+b{L};
            if L<nl, a{L+1}=tanh(z{L}); else, a{L+1}=z{L}; end
        end
        dz = 2*(a{nl+1}-Tb(j));
        for L=nl:-1:1
            gW{L}=gW{L}+dz*a{L}'; gb{L}=gb{L}+dz;
            if L>1, dz = (W{L}'*dz).*(1-a{L}.^2); end
        end
    end
    nu = feedforwardnet(arch); nu=configure(nu,Xc,Tc);
    nu.IW{1,1}=W{1}-alpha*gW{1}; nu.b{1}=b{1}-alpha*gb{1};
    for L=2:nl, nu.LW{L,L-1}=W{L}-alpha*gW{L}; nu.b{L}=b{L}-alpha*gb{L}; end
    nu.trainParam.epochs=30; nu.trainParam.goal=1e-8; nu.trainParam.showWindow=false;
    nu=train(nu,Xc,Tc);
    mse = mean((Tc-nu(Xc)).^2);
end
