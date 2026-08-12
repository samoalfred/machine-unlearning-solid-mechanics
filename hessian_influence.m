%% Hessian-based (Newton) influence unlearning, WITHOUT fine-tuning.
% Tests whether a correctly-scaled second-order influence update removes the
% influence of corrupted data on its own. The removal direction is
%     theta_unlearned = theta_hat + (H + lambda I)^{-1} g_bad,
% where g_bad is the corrupted subset's contribution to the total mean-loss
% gradient and H is the Hessian of the total mean loss at theta_hat. The
% linear system is solved by conjugate gradient using Hessian-vector products
% computed by automatic differentiation (double backprop) -- H is never formed.
%
% Compared against: biased model, retrained-from-scratch (clean only), and the
% inert first-order step used previously. No fine-tuning is applied to any
% influence result.

clear; clc; close all;
rng(12);
k=100; Fmin=0; Fmax=500; Nclean=1000; Nbad=50; Ntot=Nclean+Nbad;
layers=[1 20 15 1];
Fsc=Fmax;                 % input scale for conditioning
nSeed=3;
lams=[1e-3 1e-2 1e-1];    % CG damping values to sweep

Fg=linspace(Fmin,Fmax,400); trueg=Fg/k;   % test grid

res_biased=zeros(nSeed,1); res_retr=zeros(nSeed,1); res_first=zeros(nSeed,1);
res_newton=zeros(nSeed,numel(lams));
fd_newton=zeros(nSeed,numel(lams));
stepnorm=zeros(nSeed,numel(lams));

for s=1:nSeed
    rng(300+s);
    F=(Fmax-Fmin)*rand(Nclean,1)+Fmin; d=F/k+0.01*randn(Nclean,1);
    Fb=(Fmax-Fmin)*rand(Nbad,1)+Fmin; db=2*(Fb/k);
    Xc=F'/Fsc; Tc=d';                    % clean (normalized input)
    Xb=Fb'/Fsc; Tb=db';                  % corrupted
    Xall=[Xc Xb]; Tall=[Tc Tb];

    % ---- train biased (all data) and retrained (clean only) with Adam ----
    pB = trainAdam(layers, Xall, Tall, 4000);
    pR = trainAdam(layers, Xc,   Tc,   4000);

    res_biased(s) = gridMSE(pB, Fg/Fsc, trueg, layers);
    res_retr(s)   = gridMSE(pR, Fg/Fsc, trueg, layers);
    fR = predict(pR, Fg/Fsc, layers);

    % ---- RHS: corrupted subset contribution to total mean-loss gradient ----
    gbad = dlfeval(@gradSubset, pB, dlarray(Xb), dlarray(Tb), Ntot, numel(layers)-1);

    % ---- first-order inert step for contrast: theta - alpha*g_bad (alpha=1e-6)
    pF = axpy(pB, -1e-6, gbad);
    res_first(s) = gridMSE(pF, Fg/Fsc, trueg, layers);

    % ---- Newton step for each damping ----
    for li=1:numel(lams)
        v = cgSolve(pB, gbad, dlarray(Xall), dlarray(Tall), lams(li), 120, 1e-6, numel(layers)-1);
        pN = axpy(pB, 1.0, v);                    % theta + (H+lam I)^{-1} g_bad
        res_newton(s,li) = gridMSE(pN, Fg/Fsc, trueg, layers);
        fd_newton(s,li)  = mean((predict(pN,Fg/Fsc,layers)-fR).^2);
        stepnorm(s,li)   = sqrt(sdot(v,v));
    end
    fprintf('seed %d: biased %.3e retrain %.3e first-order %.3e | newton(lam) ', ...
        s, res_biased(s), res_retr(s), res_first(s));
    fprintf('%.3e ', res_newton(s,:)); fprintf('\n');
end

fprintf('\n=== Hessian/Newton influence unlearning (mean over %d seeds), NO fine-tuning ===\n', nSeed);
fprintf('Biased MSE-to-true      : %.3e\n', mean(res_biased));
fprintf('Retrained MSE-to-true   : %.3e\n', mean(res_retr));
fprintf('First-order step (inert): %.3e\n', mean(res_first));
for li=1:numel(lams)
    fprintf('Newton lam=%.0e: MSE %.3e | func-dist-to-retrained %.3e | mean||step|| %.3e\n', ...
        lams(li), mean(res_newton(:,li)), mean(fd_newton(:,li)), mean(stepnorm(:,li)));
end

if ~exist('results','dir'); mkdir('results'); end
hi=struct('biased',mean(res_biased),'retrained',mean(res_retr),'first_order',mean(res_first), ...
    'lams',lams,'newton_mse',mean(res_newton,1),'newton_fd',mean(fd_newton,1),'stepnorm',mean(stepnorm,1));
save(fullfile('results','hessian_influence.mat'),'hi');
fprintf('\nSaved results/hessian_influence.mat\n');

%% ================= helpers =================
function Y=mlp(X,p,L)
    A=X;
    for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L));
end
function p=initP(layers)
    p=struct;
    for i=1:numel(layers)-1
        p.(sprintf('W%d',i))=dlarray(sqrt(2/layers(i))*randn(layers(i+1),layers(i)));
        p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1));
    end
end
function p=trainAdam(layers,X,T,iters)
    p=initP(layers); Xd=dlarray(X); Td=dlarray(T); aG=[]; aS=[]; lr=5e-3; nL=numel(layers)-1;
    for it=1:iters
        [~,g]=dlfeval(@lossGrad,p,Xd,Td,nL);
        if it==round(0.6*iters), lr=1e-3; end
        if it==round(0.85*iters), lr=2e-4; end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr);
    end
end
function [loss,grad]=lossGrad(p,X,T,nL)
    loss=mean((mlp(X,p,nL)-T).^2,'all'); grad=dlgradient(loss,p);
end
function g=gradSubset(p,Xsub,Tsub,Ntot,nL)
    % contribution of the subset to the TOTAL mean loss: (1/Ntot) sum_sub (f-T)^2
    L=sum((mlp(Xsub,p,nL)-Tsub).^2,'all')/Ntot;
    g=dlgradient(L,p);
end
function Hv=hvpFun(p,v,X,T,nL)
    g=dlgradient(mean((mlp(X,p,nL)-T).^2,'all'), p, 'EnableHigherDerivatives',true);
    gv=0; f=fieldnames(g);
    for i=1:numel(f), gv=gv+sum(g.(f{i}).*v.(f{i}),'all'); end
    Hv=dlgradient(gv,p);
end
function v=cgSolve(p,g,X,T,lam,maxit,tol,nL)
    x=scal(g,0); r=g; pp=r; rs=sdot(r,r); g0=sqrt(rs);
    for it=1:maxit
        Hp=dlfeval(@hvpFun,p,pp,X,T,nL);
        Ap=axpy(Hp,lam,pp);                 % (H + lam I) pp
        denom=sdot(pp,Ap); if denom<=0, denom=denom+lam*sdot(pp,pp)+1e-12; end
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
% struct-vector ops
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end
function o=scal(a,c);    o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end
function s=sdot(a,b);    s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end
function y=predict(p,X,layers); y=extractdata(mlp(dlarray(X),p,numel(layers)-1)); end
function m=gridMSE(p,X,truey,layers); y=predict(p,X,layers); m=mean((y-truey).^2); end
