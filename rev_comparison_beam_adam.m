%% Revision: Table 10 rerun -- beam case study, all methods in the Adam / AD framework.
%% Compares, from the same biased model and the same data per seed:
%%   (1) full retraining on the clean data,
%%   (2) SISA: 5 shards; the shards that contain corrupted points are retrained
%%       without them and the shard predictions are averaged,
%%   (3) Fisher forgetting: a diagonal-Fisher-scaled scrub step on the parameters,
%%       followed by the same fine-tune as Algorithm 1 (step size chosen per seed
%%       from a small grid by the retained-data loss -- no test data used),
%%   (4) Algorithm 1: damped Newton influence step + the same fine-tune.
%% Same networks, optimizer, and budgets as the main results (algorithm2_ff.m).
%% Reports clean-test MSE and wall-clock time / speedup over full retraining.
%% Writes results/rev_comparison_beam_adam.mat.
clear; clc; close all;
layers=[1 20 15 1]; nL=numel(layers)-1; p=1;      % beam (Case Study 1)
S=5; adamIters=1500; ftIters=250; cgIters=60; gammaFrac=1e-3;
nShard=5; alphaGrid=[1e-6 1e-5 1e-4 1e-3];
MSE=zeros(S,4); TIME=zeros(S,4); alphaUsed=zeros(S,1);   % columns: retrain, SISA, Fisher, Algorithm 1
for s=1:S
    rng(2000+s);
    [Xc,Tc,Xb,Tb,Xt,Tt]=genFF(p); Ntot=size(Xc,2)+size(Xb,2);
    Xall=dlarray([Xc Xb]); Tall=dlarray([Tc Tb]); Xcd=dlarray(Xc); Tcd=dlarray(Tc);
    pB=trainAdam(layers,Xall,Tall,adamIters);                     % deployed (biased) model

    % (1) full retraining on clean data
    tic; pR=trainAdam(layers,Xcd,Tcd,adamIters); TIME(s,1)=toc; MSE(s,1)=mseT(pR,Xt,Tt,nL);

    % (2) SISA: shard the full training set, train one model per shard (deployment, not timed),
    %     then retrain only the shards that contain corrupted points, without those points
    Xf=[Xc Xb]; Tf=[Tc Tb]; isBad=[false(1,size(Xc,2)) true(1,size(Xb,2))];
    idx=randperm(Ntot); edges=round(linspace(0,Ntot,nShard+1)); shardP=cell(nShard,1);
    for k=1:nShard, sel=idx(edges(k)+1:edges(k+1)); shardP{k}=trainAdam(layers,dlarray(Xf(:,sel)),dlarray(Tf(:,sel)),adamIters); end
    tic;
    for k=1:nShard, sel=idx(edges(k)+1:edges(k+1));
        if any(isBad(sel)), keep=sel(~isBad(sel)); shardP{k}=trainAdam(layers,dlarray(Xf(:,keep)),dlarray(Tf(:,keep)),adamIters); end
    end
    TIME(s,2)=toc;
    ys=0; for k=1:nShard, ys=ys+pred(shardP{k},Xt,nL); end; ys=ys/nShard; MSE(s,2)=mean((ys(:)-Tt(:)).^2);

    % (3) Fisher forgetting: diagonal empirical Fisher from the retained-data gradient,
    %     scrub along the corrupted-data gradient, then the same fine-tune
    gR=dlfeval(@lgGrad,pB,Xcd,Tcd,nL); Fd=sqField(gR);
    gbad=dlfeval(@gradSubset,pB,dlarray(Xb),dlarray(Tb),Ntot,nL);
    best=inf;
    for a=alphaGrid       % pick the step by the retained-data loss only
        pF=fineTune(fisherStep(pB,gbad,Fd,a),Xcd,Tcd,ftIters,nL); L=mseT(pF,Xc,Tc,nL);
        if isfinite(L) && L<best, best=L; alphaUsed(s)=a; end
    end
    tic; gR=dlfeval(@lgGrad,pB,Xcd,Tcd,nL); Fd=sqField(gR); gbad=dlfeval(@gradSubset,pB,dlarray(Xb),dlarray(Tb),Ntot,nL);
    pF=fineTune(fisherStep(pB,gbad,Fd,alphaUsed(s)),Xcd,Tcd,ftIters,nL); TIME(s,3)=toc; MSE(s,3)=mseT(pF,Xt,Tt,nL);

    % (4) Algorithm 1: damped Newton influence step + the same fine-tune
    tic; gbad=dlfeval(@gradSubset,pB,dlarray(Xb),dlarray(Tb),Ntot,nL);
    lam=powerLambda(pB,Xall,Tall,nL,25); v=cgSolve(pB,gbad,Xall,Tall,gammaFrac*lam,cgIters,1e-6,nL);
    pA=fineTune(axpy(pB,1.0,v),Xcd,Tcd,ftIters,nL); TIME(s,4)=toc; MSE(s,4)=mseT(pA,Xt,Tt,nL);

    fprintf('seed %d: retrain %.2e  SISA %.2e  Fisher %.2e (alpha %.0e)  Alg1 %.2e | times %.2f %.2f %.2f %.2f s\n', ...
        s,MSE(s,1),MSE(s,2),MSE(s,3),alphaUsed(s),MSE(s,4),TIME(s,:));
end
SPD=TIME(:,1)./TIME;   % per-run speedup over full retraining
if ~exist('results','dir'); mkdir('results'); end
CB=struct('methods',{{'Retraining','SISA','Fisher forgetting','Algorithm 1 (Newton + fine-tune)'}}, ...
    'MSE',MSE,'TIME',TIME,'speedup',SPD,'alphaUsed',alphaUsed,'nShard',nShard,'S',S);
save(fullfile('results','rev_comparison_beam_adam.mat'),'CB');
fprintf('\n=== Beam comparison, Adam/AD framework, mean over %d seeds ===\n',S);
for j=1:4, fprintf('%-34s MSE %.2e +- %.1e   speedup %.1fx\n',CB.methods{j},mean(MSE(:,j)),std(MSE(:,j)),mean(SPD(:,j))); end
fprintf('Saved results/rev_comparison_beam_adam.mat\n');

%% ---- helpers specific to this script ----
function g=lgGrad(p,X,T,nL); l=mean((mlp(X,p,nL)-T).^2,'all'); g=dlgradient(l,p); end
function F=sqField(g); F=g; f=fieldnames(g); for i=1:numel(f), F.(f{i})=g.(f{i}).^2; end, end
function o=fisherStep(p,g,Fdiag,al); o=p; f=fieldnames(p);
    for i=1:numel(f), o.(f{i})=p.(f{i})+al*g.(f{i})./(Fdiag.(f{i})+1e-5); end, end

%% ---- shared helpers (identical to algorithm2_ff.m) ----
% genFF: analytic data generators per case. Each returns clean (Xc,Tc), a
% corrupted subset (Xb,Tb) with a case-specific bias, and a clean test set (Xt,Tt).
function [Xc,Tc,Xb,Tb,Xt,Tt]=genFF(p)
    switch p
      case 1   % beam deflection: linear law d=F/k; corruption doubles the slope
        k=100; Fc=500*rand(1,1000); Fb=500*rand(1,50);
        Xc=Fc/500; Tc=Fc/k+0.01*randn(1,1000); Xb=Fb/500; Tb=2*Fb/k;
        Fg=linspace(0,500,400); Xt=Fg/500; Tt=Fg/k;
      case 2   % temperature-dependent modulus; corruption shifts temperature by +20
        E0=200; al=0.005; Tc0=100*rand(1,800); Sc=50*rand(1,800);
        Xc=[Tc0/100; Sc/50]; Tc=Sc./(E0*(1-al*Tc0))+0.001*randn(1,800);
        Tb0=100*rand(1,30); Sb=50*rand(1,30); Xb=[(Tb0+20)/100; Sb/50]; Tb=Sb./(E0*(1-al*Tb0));
        Tg=100*rand(1,400); Sg=50*rand(1,400); Xt=[Tg/100; Sg/50]; Tt=Sg./(E0*(1-al*Tg));
      case 3   % multi-axial stress (2 outputs); corruption cross-couples the channels
        E=70; G=26.3; sx=100*rand(1,600); tx=50*rand(1,600);
        Xc=[sx/100; tx/50]; Tc=[sx/E+1e-4*randn(1,600); tx/G+1e-4*randn(1,600)];
        sb=100*rand(1,40); tb=50*rand(1,40); Xb=[sb/100; tb/50]; Tb=[sb/E+0.1*tb/G; tb/G+0.05*sb/E];
        sg=100*rand(1,400); tg=50*rand(1,400); Xt=[sg/100; tg/50]; Tt=[sg/E; tg/G];
      case 4   % cyclic-loading damage; corruption inflates damage by 1.5x
        c=1e-12; m=3.5; N=1+9999*rand(1,500); Sn=100+200*rand(1,500);
        Dc=1-exp(-c*N.*Sn.^m); Xc=[N/1e4; Sn/300]; Tc=min(1,max(0,Dc+0.02*randn(1,500)));
        Nb=1+9999*rand(1,50); Sb=100+200*rand(1,50); Xb=[Nb/1e4; Sb/300]; Tb=min(1,max(0,1.5*(1-exp(-c*Nb.*Sb.^m))));
        Ng=1+9999*rand(1,400); Sg=100+200*rand(1,400); Xt=[Ng/1e4; Sg/300]; Tt=1-exp(-c*Ng.*Sg.^m);
    end
end
% mlp: forward pass -- tanh hidden layers, linear output
function Y=mlp(X,p,L); A=X; for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L)); end
% initP: He-style weight init, zero biases, stored as dlarrays
function p=initP(layers); p=struct;
    for i=1:numel(layers)-1, p.(sprintf('W%d',i))=dlarray(sqrt(2/layers(i))*randn(layers(i+1),layers(i)));
        p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1)); end
end
% trainAdam: Adam loop with a stepped learning rate (5e-3 -> 1e-3 at 60% -> 2e-4 at 85%)
function p=trainAdam(layers,X,T,iters); p=initP(layers); aG=[];aS=[];lr=5e-3;nL=numel(layers)-1;
    for it=1:iters, [~,g]=dlfeval(@lg,p,X,T,nL); if it==round(0.6*iters),lr=1e-3;end; if it==round(0.85*iters),lr=2e-4;end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
% fineTune: like trainAdam but starts from the given params (1e-3 -> 2e-4 at 70%)
function p=fineTune(p,X,T,iters,nL); aG=[];aS=[];lr=1e-3;
    for it=1:iters, [~,g]=dlfeval(@lg,p,X,T,nL); if it==round(0.7*iters),lr=2e-4;end; [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
% lg: mean-squared loss and its gradient
function [l,g]=lg(p,X,T,nL); l=mean((mlp(X,p,nL)-T).^2,'all'); g=dlgradient(l,p); end
% gradSubset: g_bad = gradient of (sum of subset squared errors)/Ntot (correctly weighted)
function g=gradSubset(p,Xs,Ts,Ntot,nL); l=sum((mlp(Xs,p,nL)-Ts).^2,'all')/Ntot; g=dlgradient(l,p); end
% hvp: Hessian-vector product via double backprop (grad of grad.v); H is never formed
function Hv=hvp(p,v,X,T,nL)
    g=dlgradient(mean((mlp(X,p,nL)-T).^2,'all'),p,'EnableHigherDerivatives',true);
    gv=0; f=fieldnames(g); for i=1:numel(f), gv=gv+sum(g.(f{i}).*v.(f{i}),'all'); end; Hv=dlgradient(gv,p);
end
% powerLambda: power iteration (through hvp) for the top Hessian eigenvalue
function lam=powerLambda(p,X,T,nL,iters); v=randLike(p); v=nrm(v);
    for it=1:iters, Hv=dlfeval(@hvp,p,v,X,T,nL); lam=sdot(v,Hv); v=nrm(Hv); end
end
% cgSolve: damped conjugate gradient for (H + lam I) v = g, with a non-positive-curvature guard
function v=cgSolve(p,g,X,T,lam,maxit,tol,nL)
    x=scal(g,0); r=g; pp=r; rs=sdot(r,r); g0=sqrt(rs);
    for it=1:maxit, Hp=dlfeval(@hvp,p,pp,X,T,nL); Ap=axpy(Hp,lam,pp);   % Ap = (H + lam I) pp
        den=sdot(pp,Ap); if den<=0, den=den+lam*sdot(pp,pp)+1e-12; end   % guard against non-positive curvature
        al=rs/den; x=axpy(x,al,pp); r=axpy(r,-al,Ap); rs2=sdot(r,r); if sqrt(rs2)<tol*g0, break; end
        pp=axpy(r,rs2/rs,pp); rs=rs2; end
    v=x;
end
% mseT/pred/pred2: evaluate a model on test inputs -> numeric MSE / predictions
function m=mseT(p,Xt,Tt,nL); Y=extractdata(mlp(dlarray(Xt),p,nL)); m=mean((Y(:)-Tt(:)).^2); end
function y=pred(p,X,nL); y=extractdata(mlp(dlarray(X),p,nL)); end
function y=pred2(p,X,nL); y=extractdata(mlp(dlarray(X),p,nL)); end
% struct-vector algebra: lets CG/power-iteration treat the parameter struct as one vector
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end   % a + al*b
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end   % c*a
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end   % inner product
function v=randLike(p); v=p; f=fieldnames(p); for i=1:numel(f), v.(f{i})=dlarray(randn(size(p.(f{i})))); end, end   % random vector shaped like p
function v=nrm(v); n=sqrt(sdot(v,v)); f=fieldnames(v); for i=1:numel(f), v.(f{i})=v.(f{i})/n; end, end   % normalize to unit length
