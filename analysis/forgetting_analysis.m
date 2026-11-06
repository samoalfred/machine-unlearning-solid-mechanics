%% Forgetting analysis: what actually drives the recovery, and is it forgetting?
% Addresses reviewer concerns that (i) the recovery may be warm-start
% fine-tuning rather than the influence step, and (ii) improvement may be the
% physics penalty "telling the model the answer" rather than removing data
% influence. Also computes forgetting metrics vs a retrained model and a
% valid-data (obsolete-state) forgetting test with NO ground-truth penalty.
%
% All numbers are computed. Beam-deflection testbed, feedforward [20 15].

clear; clc; close all;
rng(1);
k=100; Fmin=0; Fmax=500; Nclean=1000; Nbad=50; arch=[20 15];
nSeed=10; alpha=1e-6;

% storage: rows = seeds, cols = conditions
% conditions: 1 biased, 2 influence-only, 3 warmFT-only, 4 full(infl+FT), 5 retrained
labels = {'Biased','Influence-only','Warm-FT-only','Full (infl+FT)','Retrained'};
mse_true = zeros(nSeed,5);   % MSE to TRUE physics on clean test
mse_badlabel = zeros(nSeed,5); % MSE to CORRUPTED labels at corrupted inputs (memorization)
func_dist_ret = zeros(nSeed,4);% functional distance to retrained model (cond 1..4)

Fg = linspace(Fmin,Fmax,400); trueg = Fg/k;   % test grid

for s=1:nSeed
    rng(100+s);
    F=(Fmax-Fmin)*rand(Nclean,1)+Fmin; d=F/k+0.01*randn(Nclean,1);
    Xc=F'; Tc=d';
    Fb=(Fmax-Fmin)*rand(Nbad,1)+Fmin; dbad=2*(Fb/k);   % corrupted labels
    Xb=Fb'; Tb=dbad';
    Xf=[Xc Xb]; Tf=[Tc Tb];

    % Biased and Retrained (gold standard: train from scratch on clean only)
    netB = mkTrain(Xf,Tf,arch,500);
    netR = mkTrain(Xc,Tc,arch,500);

    % gradient on corrupted points (manual backprop, 3 weight layers)
    [W,b] = getWB(netB,arch);
    g = accumGrad(W,b,Xb,Tb);

    % (2) influence-only: subtract alpha*g, no fine-tune
    net2 = setWB(Xc,Tc,arch, sub(W,g,alpha,1), sub(b,g,alpha,0));

    % (3) warm-FT-only: biased weights, fine-tune on clean data, NO influence step
    net3 = setWBtrain(Xc,Tc,arch,W,b,30);

    % (4) full: influence step then fine-tune on clean data
    net4 = setWBtrain(Xc,Tc,arch, sub(W,g,alpha,1), sub(b,g,alpha,0), 30);

    nets = {netB,net2,net3,net4,netR};
    fR = netR(Fg);
    for c=1:5
        y = nets{c}(Fg);
        mse_true(s,c) = mean((y-trueg).^2);
        yb = nets{c}(Xb);
        mse_badlabel(s,c) = mean((yb-Tb).^2);   % low => still fits corrupted labels
        if c<=4, func_dist_ret(s,c) = mean((nets{c}(Fg)-fR).^2); end
    end
end

fprintf('\n=== Decomposition (beam), mean over %d seeds ===\n', nSeed);
fprintf('%-16s  MSE-to-true    MSE-to-badlabel   FuncDist-to-Retrained\n','Condition');
for c=1:5
    if c<=4, fd=sprintf('%.3e',mean(func_dist_ret(:,c))); else, fd='   -  (is retrained)'; end
    fprintf('%-16s  %.3e      %.3e         %s\n', labels{c}, mean(mse_true(:,c)), mean(mse_badlabel(:,c)), fd);
end

%% Valid-data (obsolete-state) forgetting: NO ground-truth penalty
% Two materials A (kA=100) and B (kB=50). Train on A+B (conflicting labels
% for same F). "Forget A" should move predictions toward a model retrained on
% B only. Physics penalty cannot help here: both A and B are valid physics.
rng(7);
kA=100; kB=50; nA=600; nB=600;
mfd_full=zeros(nSeed,1); mfd_unlearn=zeros(nSeed,1);
for s=1:nSeed
    rng(200+s);
    FA=(Fmax-Fmin)*rand(nA,1)+Fmin; dA=FA/kA+0.01*randn(nA,1);
    FB=(Fmax-Fmin)*rand(nB,1)+Fmin; dB=FB/kB+0.01*randn(nB,1);
    XA=FA'; TA=dA'; XB=FB'; TB2=dB';
    Xall=[XA XB]; Tall=[TA TB2];

    netFull = mkTrain(Xall,Tall,arch,500);     % trained on A+B
    netBonly = mkTrain(XB,TB2,arch,500);       % retrained on B only (target)

    [W,b]=getWB(netFull,arch);
    gA = accumGrad(W,b,XA,TA);                 % gradient on A (data to forget)
    netUn = setWBtrain(XB,TB2,arch, sub(W,gA,alpha,1), sub(b,gA,alpha,0), 30); % forget A, fine-tune on B

    fB = netBonly(Fg);
    mfd_full(s)    = mean((netFull(Fg)-fB).^2);   % how far A+B model is from B-only
    mfd_unlearn(s) = mean((netUn(Fg)-fB).^2);     % how far unlearned model is from B-only
end
fprintf('\n=== Valid-data forgetting (forget material A) ===\n');
fprintf('Functional distance to retrained-on-B:  A+B model %.3e  ->  unlearned %.3e  (%.0f%% closer)\n', ...
    mean(mfd_full), mean(mfd_unlearn), (1-mean(mfd_unlearn)/mean(mfd_full))*100);

%% Save
if ~exist('results','dir'); mkdir('results'); end
fa=struct('labels',{labels},'mse_true',mean(mse_true),'mse_badlabel',mean(mse_badlabel), ...
    'func_dist_ret',mean(func_dist_ret), ...
    'valid_full',mean(mfd_full),'valid_unlearn',mean(mfd_unlearn));
save(fullfile('results','forgetting.mat'),'fa');
fprintf('\nSaved results/forgetting.mat\n');

%% ---- helpers ----
function net=mkTrain(X,T,arch,ep)
    net=feedforwardnet(arch); net.trainParam.epochs=ep; net.trainParam.goal=1e-8;
    net.trainParam.showWindow=false; net=configure(net,X,T); net=train(net,X,T);
end
function [W,b]=getWB(net,arch)
    nl=numel(arch)+1; W=cell(nl,1); b=cell(nl,1);
    W{1}=net.IW{1,1}; b{1}=net.b{1};
    for L=2:nl, W{L}=net.LW{L,L-1}; b{L}=net.b{L}; end
end
function g=accumGrad(W,b,X,T)
    nl=numel(W); g=cellfun(@(w) zeros(size(w)),W,'uni',0);
    for j=1:numel(T)
        a=cell(nl+1,1); a{1}=X(j);
        for L=1:nl, z=W{L}*a{L}+b{L}; if L<nl, a{L+1}=tanh(z); else, a{L+1}=z; end, end
        dz=2*(a{nl+1}-T(j));
        for L=nl:-1:1
            g{L}=g{L}+dz*a{L}';
            if L>1, dz=(W{L}'*dz).*(1-a{L}.^2); end
        end
    end
    % note: bias grads folded via same dz; return weight grads only (bias handled in sub)
end
function out=sub(W,g,al,isW)
    % returns W - al*g for weights (isW=1). For biases (isW=0) we approximate the
    % bias gradient as zero-order via the same routine is omitted; biases are
    % left unchanged by the influence step (dominant effect is in weights).
    if isW
        out=cellfun(@(w,gg) w-al*gg, W, g, 'uni',0);
    else
        out=W; % biases unchanged by influence step
    end
end
function net=setWB(X,T,arch,W,b)
    net=feedforwardnet(arch); net=configure(net,X,T); nl=numel(arch)+1;
    net.IW{1,1}=W{1}; net.b{1}=b{1};
    for L=2:nl, net.LW{L,L-1}=W{L}; net.b{L}=b{L}; end
end
function net=setWBtrain(X,T,arch,W,b,ep)
    net=setWB(X,T,arch,W,b);
    net.trainParam.epochs=ep; net.trainParam.goal=1e-8; net.trainParam.showWindow=false;
    net=train(net,X,T);
end
