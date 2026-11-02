%% Resumable copy of hessian_influence.m (results are identical).
% Hessian-based (Newton) influence unlearning, WITHOUT fine-tuning, on the beam.
% Compares: biased, retrained (clean), inert first-order step, and a damped
% Newton step over a sweep of CG damping values. See hessian_influence.m for the
% full method notes.
%
% RESUMABLE: state is checkpointed per completed seed to
% results/hessian_influence_partial.mat; each call advances up to seedsPerCall
% seeds and then stops, exactly like algorithm2_ff.m / algorithm2_pinn.m. Because
% every seed re-seeds the RNG deterministically with rng(300+s) and is fully
% self-contained, splitting the seed loop across invocations reproduces the
% originals bit-for-bit; the seed-averaged summary is unchanged. Re-run until the
% final summary prints. Delete the *_partial.mat file to start over.

clear; clc; close all;
rng(12);

% ---- problem setup (identical to hessian_influence.m) ----
k=100; Fmin=0; Fmax=500; Nclean=1000; Nbad=50; Ntot=Nclean+Nbad;
layers=[1 20 15 1];
Fsc=Fmax;                 % input scale for conditioning
nSeed=3;
lams=[1e-3 1e-2 1e-1];    % CG damping values to sweep
seedsPerCall=1;           % seeds advanced per invocation (resumability cap)

Fg=linspace(Fmin,Fmax,400); trueg=Fg/k;   % test grid

% ---- resume from checkpoint (or start fresh) ----
if ~exist('results','dir'); mkdir('results'); end
pf=fullfile('results','hessian_influence_partial.mat');   % per-seed checkpoint
if exist(pf,'file'), Lp=load(pf); A=Lp.A;
else, A=struct('biased',[],'retr',[],'first',[],'newton',[],'fd',[],'stepnorm',[]); end

% ---- per-seed loop (runs up to seedsPerCall seeds, then checkpoints) ----
ran=0;
while size(A.biased,1)<nSeed && ran<seedsPerCall
    s=size(A.biased,1)+1; ran=ran+1;
    rng(300+s);           % deterministic per-seed RNG (makes the seed self-contained)
    F=(Fmax-Fmin)*rand(Nclean,1)+Fmin; d=F/k+0.01*randn(Nclean,1);
    Fb=(Fmax-Fmin)*rand(Nbad,1)+Fmin; db=2*(Fb/k);
    Xc=F'/Fsc; Tc=d';                    % clean (normalized input)
    Xb=Fb'/Fsc; Tb=db';                  % corrupted
    Xall=[Xc Xb]; Tall=[Tc Tb];          % full training set

    % train biased (all data) and retrained (clean only)
    pB = trainAdam(layers, Xall, Tall, 4000);
    pR = trainAdam(layers, Xc,   Tc,   4000);

    resB = gridMSE(pB, Fg/Fsc, trueg, layers);
    resR = gridMSE(pR, Fg/Fsc, trueg, layers);
    fR   = predict(pR, Fg/Fsc, layers);   % retrained predictions (function-distance ref)

    % RHS: corrupted subset contribution to total mean-loss gradient
    gbad = dlfeval(@gradSubset, pB, dlarray(Xb), dlarray(Tb), Ntot, numel(layers)-1);

    % first-order inert step for contrast: theta - alpha*g_bad (alpha=1e-6)
    pF = axpy(pB, -1e-6, gbad);
    resF = gridMSE(pF, Fg/Fsc, trueg, layers);

    % Newton step for each damping
    newtRow=zeros(1,numel(lams)); fdRow=zeros(1,numel(lams)); snRow=zeros(1,numel(lams));
    for li=1:numel(lams)
        v = cgSolve(pB, gbad, dlarray(Xall), dlarray(Tall), lams(li), 120, 1e-6, numel(layers)-1);
        pN = axpy(pB, 1.0, v);                    % theta + (H+lam I)^{-1} g_bad
        newtRow(li) = gridMSE(pN, Fg/Fsc, trueg, layers);
        fdRow(li)   = mean((predict(pN,Fg/Fsc,layers)-fR).^2);   % distance to retrained
        snRow(li)   = sqrt(sdot(v,v));
    end

    % append this seed's results and checkpoint
    A.biased(end+1,1)=resB; A.retr(end+1,1)=resR; A.first(end+1,1)=resF;
    A.newton(end+1,:)=newtRow; A.fd(end+1,:)=fdRow; A.stepnorm(end+1,:)=snRow;
    save(pf,'A');
    fprintf('seed %d: biased %.3e retrain %.3e first-order %.3e | newton(lam) ', s, resB, resR, resF);
    fprintf('%.3e ', newtRow); fprintf('\n');
end

% ---- finalize once all seeds are done ----
if size(A.biased,1)>=nSeed
    res_biased=A.biased; res_retr=A.retr; res_first=A.first;
    res_newton=A.newton; fd_newton=A.fd; stepnorm=A.stepnorm;
    fprintf('\n=== Hessian/Newton influence unlearning (mean over %d seeds), NO fine-tuning ===\n', nSeed);
    fprintf('Biased MSE-to-true      : %.3e\n', mean(res_biased));
    fprintf('Retrained MSE-to-true   : %.3e\n', mean(res_retr));
    fprintf('First-order step (inert): %.3e\n', mean(res_first));
    for li=1:numel(lams)
        fprintf('Newton lam=%.0e: MSE %.3e | func-dist-to-retrained %.3e | mean||step|| %.3e\n', ...
            lams(li), mean(res_newton(:,li)), mean(fd_newton(:,li)), mean(stepnorm(:,li)));
    end
    hi=struct('biased',mean(res_biased),'retrained',mean(res_retr),'first_order',mean(res_first), ...
        'lams',lams,'newton_mse',mean(res_newton,1),'newton_fd',mean(fd_newton,1),'stepnorm',mean(stepnorm,1));
    save(fullfile('results','hessian_influence.mat'),'hi');
    fprintf('\nSaved results/hessian_influence.mat\n');
else
    fprintf('Progress: %d/%d seeds done. Re-run to continue.\n', size(A.biased,1), nSeed);
end

%% ================= helpers (verbatim from hessian_influence.m) =================
% mlp: forward pass -- tanh hidden layers, linear output
function Y=mlp(X,p,L)
    A=X;
    for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L));
end
% initP: He-style weight init, zero biases, as dlarrays
function p=initP(layers)
    p=struct;
    for i=1:numel(layers)-1
        p.(sprintf('W%d',i))=dlarray(sqrt(2/layers(i))*randn(layers(i+1),layers(i)));
        p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1));
    end
end
% trainAdam: Adam loop with a stepped learning rate (5e-3 -> 1e-3 at 60% -> 2e-4 at 85%)
function p=trainAdam(layers,X,T,iters)
    p=initP(layers); Xd=dlarray(X); Td=dlarray(T); aG=[]; aS=[]; lr=5e-3; nL=numel(layers)-1;
    for it=1:iters
        [~,g]=dlfeval(@lossGrad,p,Xd,Td,nL);
        if it==round(0.6*iters), lr=1e-3; end
        if it==round(0.85*iters), lr=2e-4; end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr);
    end
end
% lossGrad: mean-squared loss and its gradient
function [loss,grad]=lossGrad(p,X,T,nL)
    loss=mean((mlp(X,p,nL)-T).^2,'all'); grad=dlgradient(loss,p);
end
% gradSubset: g_bad = gradient of the subset's share of the TOTAL mean loss
function g=gradSubset(p,Xsub,Tsub,Ntot,nL)
    L=sum((mlp(Xsub,p,nL)-Tsub).^2,'all')/Ntot;
    g=dlgradient(L,p);
end
% hvpFun: Hessian-vector product via double backprop (grad of grad.v); H is never formed
function Hv=hvpFun(p,v,X,T,nL)
    g=dlgradient(mean((mlp(X,p,nL)-T).^2,'all'), p, 'EnableHigherDerivatives',true);
    gv=0; f=fieldnames(g);
    for i=1:numel(f), gv=gv+sum(g.(f{i}).*v.(f{i}),'all'); end
    Hv=dlgradient(gv,p);
end
% cgSolve: damped conjugate gradient for (H + lam I) v = g, with a curvature guard
function v=cgSolve(p,g,X,T,lam,maxit,tol,nL)
    x=scal(g,0); r=g; pp=r; rs=sdot(r,r); g0=sqrt(rs);
    for it=1:maxit
        Hp=dlfeval(@hvpFun,p,pp,X,T,nL);
        Ap=axpy(Hp,lam,pp);                 % (H + lam I) pp
        denom=sdot(pp,Ap); if denom<=0, denom=denom+lam*sdot(pp,pp)+1e-12; end   % guard against non-positive curvature
        al=rs/denom;
        x=axpy(x,al,pp);
        r=axpy(r,-al,Ap);
        rs2=sdot(r,r);
        if sqrt(rs2)<tol*g0, break; end
        pp=axpy(r,rs2/rs,pp);
        rs=rs2;
    end
    v=x;
end
% struct-vector ops: treat the parameter struct as one vector
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end   % a + al*b
function o=scal(a,c);    o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end   % c*a
function s=sdot(a,b);    s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end   % inner product
% predict/gridMSE: model predictions and MSE against the true curve on the test grid
function y=predict(p,X,layers); y=extractdata(mlp(dlarray(X),p,numel(layers)-1)); end
function m=gridMSE(p,X,truey,layers); y=predict(p,X,layers); m=mean((y-truey).^2); end
