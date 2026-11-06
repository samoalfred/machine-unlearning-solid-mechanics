%% Drop-in template: run the Algorithm 1 unlearning pipeline on a REAL dataset.
%% ----------------------------------------------------------------------------
%% This script is deliberately data-agnostic. Supply a real experimental or
%% simulation dataset (e.g. DIC displacement fields, an FE database, or fatigue
%% test records) as a CSV, mark which rows are the corrupted/obsolete points to
%% remove, and it trains a surrogate, applies the damped Newton influence step
%% plus a physics-agnostic warm-start fine-tune, and reports the clean,
%% biased, and unlearned test errors exactly as in the paper's Tables 1-6.
%%
%% NO synthetic data are generated here. Nothing is hard-coded. If you have not
%% provided a dataset, the script stops with an explanatory message rather than
%% inventing data.
%% ----------------------------------------------------------------------------
%% Expected CSV format (header row required):
%%   x1, x2, ..., xd, y, is_removed
%%   - x1..xd : input features (any number of columns)
%%   - y      : scalar target (extend to multi-output if needed)
%%   - is_removed : 1 for a point to unlearn (corrupted/obsolete), 0 to retain
%% A separate clean test set (same columns minus is_removed) may be supplied as
%% CSV_TEST; if omitted, a held-out split of the retained data is used.
%% ----------------------------------------------------------------------------
clear; clc; close all;
CSV       = 'real_dataset.csv';      % <-- point this at your real data file
CSV_TEST  = '';                      % optional held-out clean test CSV
hidden    = [20 15];                 % surrogate architecture (tune per problem)
adamIters = 2000; ftIters = 50; cgIters = 60; gammaFrac = 1e-3; testFrac = 0.2;

if ~isfile(CSV)
    error(['No dataset found at "%s". This template does not fabricate data. ' ...
           'Provide a real CSV (see header comments) and re-run.'], CSV);
end
Tbl = readtable(CSV);
assert(ismember('y', Tbl.Properties.VariableNames) && ...
       ismember('is_removed', Tbl.Properties.VariableNames), ...
       'CSV must contain columns "y" and "is_removed".');
xcols = setdiff(Tbl.Properties.VariableNames, {'y','is_removed'}, 'stable');
X = Tbl{:, xcols}'; y = Tbl.y'; rm = logical(Tbl.is_removed');
d = size(X,1); layers = [d hidden 1]; nL = numel(layers)-1;

% normalize inputs/targets to unit scale (store scales for reporting)
xmu = mean(X,2); xsd = std(X,0,2)+eps; Xn = (X - xmu)./xsd;
ymu = mean(y);   ysd = std(y)+eps;     yn = (y - ymu)/ysd;

% test set
if ~isempty(CSV_TEST) && isfile(CSV_TEST)
    Tt = readtable(CSV_TEST); Xt = ((Tt{:,xcols}') - xmu)./xsd; yt = (Tt.y' - ymu)/ysd;
    retainMask = ~rm;
else
    retain = find(~rm); rng(0); perm = retain(randperm(numel(retain)));
    nTe = max(1, round(testFrac*numel(perm))); teIdx = perm(1:nTe);
    Xt = Xn(:,teIdx); yt = yn(teIdx);
    retainMask = ~rm; retainMask(teIdx) = false;   % exclude test points from training
end

Xall = dlarray(Xn);           Tall = dlarray(yn);            % biased: all data
Xcln = dlarray(Xn(:,retainMask)); Tcln = dlarray(yn(retainMask)); % retained (clean) only
Ntot = size(Xn,2);

pB = trainAdam(layers, Xall, Tall, adamIters);   % biased (trained on everything)
pR = trainAdam(layers, Xcln, Tcln, adamIters);   % retrained gold standard (clean only)

g   = dlfeval(@gradSubset, pB, dlarray(Xn(:,rm)), dlarray(yn(rm)), Ntot, nL);
lam = powerLambda(pB, Xall, Tall, nL, 25); gam = gammaFrac*lam;
v   = cgSolve(pB, g, Xall, Tall, gam, cgIters, 1e-6, nL);
pN  = axpy(pB, 1.0, v);
pA  = fineTune(pN, Xcln, Tcln, ftIters, nL);     % unlearned

mseB = mseT(pB,Xt,yt,nL); mseR = mseT(pR,Xt,yt,nL); mseA = mseT(pA,Xt,yt,nL);
errRed = (mseB-mseA)/mseB*100;
fprintf('\n=== Algorithm 1 on real dataset "%s" (%d features, %d points, %d removed) ===\n', ...
        CSV, d, Ntot, nnz(rm));
fprintf('Clean/Retrained MSE : %.3e\n', mseR);
fprintf('Biased MSE          : %.3e\n', mseB);
fprintf('Unlearned MSE       : %.3e  (error reduction %.1f%% vs biased)\n', mseA, errRed);
if ~exist('results','dir'); mkdir('results'); end
save(fullfile('results','real_data_unlearning.mat'), 'mseR','mseB','mseA','errRed','xcols');
fprintf('Saved results/real_data_unlearning.mat\n');

%% ---- helpers (identical machinery to algorithm2_ff.m) ----
function Y=mlp(X,p,L); A=X; for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L)); end
function p=initP(layers); p=struct;
    for i=1:numel(layers)-1, p.(sprintf('W%d',i))=dlarray(sqrt(2/layers(i))*randn(layers(i+1),layers(i)));
        p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1)); end
end
function p=trainAdam(layers,X,T,iters); p=initP(layers); aG=[];aS=[];lr=5e-3;nL=numel(layers)-1;
    for it=1:iters, [~,g]=dlfeval(@lg,p,X,T,nL); if it==round(0.6*iters),lr=1e-3;end; if it==round(0.85*iters),lr=2e-4;end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
function p=fineTune(p,X,T,iters,nL); if iters==0, return; end
    aG=[];aS=[];lr=1e-3;
    for it=1:iters, [~,g]=dlfeval(@lg,p,X,T,nL); if it==round(0.7*iters),lr=2e-4;end; [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
function [l,g]=lg(p,X,T,nL); l=mean((mlp(X,p,nL)-T).^2,'all'); g=dlgradient(l,p); end
function g=gradSubset(p,Xs,Ts,Ntot,nL); l=sum((mlp(Xs,p,nL)-Ts).^2,'all')/Ntot; g=dlgradient(l,p); end
function Hv=hvp(p,v,X,T,nL)
    g=dlgradient(mean((mlp(X,p,nL)-T).^2,'all'),p,'EnableHigherDerivatives',true);
    gv=0; f=fieldnames(g); for i=1:numel(f), gv=gv+sum(g.(f{i}).*v.(f{i}),'all'); end; Hv=dlgradient(gv,p);
end
function lam=powerLambda(p,X,T,nL,iters); v=randLike(p); v=nrm(v);
    for it=1:iters, Hv=dlfeval(@hvp,p,v,X,T,nL); lam=sdot(v,Hv); v=nrm(Hv); end
end
function v=cgSolve(p,g,X,T,lam,maxit,tol,nL)
    x=scal(g,0); r=g; pp=r; rs=sdot(r,r); g0=sqrt(rs);
    for it=1:maxit, Hp=dlfeval(@hvp,p,pp,X,T,nL); Ap=axpy(Hp,lam,pp);
        den=sdot(pp,Ap); if den<=0, den=den+lam*sdot(pp,pp)+1e-12; end
        al=rs/den; x=axpy(x,al,pp); r=axpy(r,-al,Ap); rs2=sdot(r,r); if sqrt(rs2)<tol*g0, break; end
        pp=axpy(r,rs2/rs,pp); rs=rs2; end
    v=x;
end
function m=mseT(p,Xt,Tt,nL); Y=extractdata(mlp(dlarray(Xt),p,nL)); m=mean((Y(:)-Tt(:)).^2); end
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end
function v=randLike(p); v=p; f=fieldnames(p); for i=1:numel(f), v.(f{i})=dlarray(randn(size(p.(f{i})))); end, end
function v=nrm(v); n=sqrt(sdot(v,v)); f=fieldnames(v); for i=1:numel(f), v.(f{i})=v.(f{i})/n; end, end
