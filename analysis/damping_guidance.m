%% Damping guidance for the Newton influence step.
% Estimates the largest eigenvalue lambda_max of the total-loss Hessian at the
% biased optimum by power iteration (using AD Hessian-vector products), so the
% working damping gamma can be expressed as a fraction of lambda_max. Beam
% problem, dlarray MLP [1 20 15 1].
clear; clc; close all;
rng(12);
k=100; Fmin=0; Fmax=500; Nclean=1000; Nbad=50; layers=[1 20 15 1]; Fsc=Fmax;
nSeed=3; gammas=[1e-3 1e-2 1e-1];
lmax=zeros(nSeed,1);
for s=1:nSeed
    rng(300+s);
    F=(Fmax-Fmin)*rand(Nclean,1)+Fmin; d=F/k+0.01*randn(Nclean,1);
    Fb=(Fmax-Fmin)*rand(Nbad,1)+Fmin; db=2*(Fb/k);
    Xall=dlarray([F' Fb']/Fsc); Tall=dlarray([d' db']);
    p=trainAdam(layers,Xall,Tall,4000);
    % power iteration for lambda_max of Hessian of total mean loss
    v=randStruct(p); v=nrm(v);
    lam=0;
    for it=1:40
        Hv=dlfeval(@hvp,p,v,Xall,Tall,numel(layers)-1);
        lam=sdot(v,Hv); v=nrm(Hv);
    end
    lmax(s)=lam;
    fprintf('seed %d: lambda_max ~ %.3e\n', s, lam);
end
Lm=mean(lmax);
fprintf('\n=== Damping guidance (beam) ===\n');
fprintf('Estimated lambda_max (mean over %d seeds): %.3e\n', nSeed, Lm);
for g=gammas
    fprintf('gamma=%.0e  ->  gamma/lambda_max = %.2e\n', g, g/Lm);
end
if ~exist('results','dir'); mkdir('results'); end
dg=struct('lambda_max',Lm,'gammas',gammas,'ratio',gammas/Lm); save(fullfile('results','damping_guidance.mat'),'dg');
fprintf('Saved results/damping_guidance.mat\n');

%% helpers
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
function [l,g]=lg(p,X,T,nL); l=mean((mlp(X,p,nL)-T).^2,'all'); g=dlgradient(l,p); end
function Hv=hvp(p,v,X,T,nL)
    g=dlgradient(mean((mlp(X,p,nL)-T).^2,'all'),p,'EnableHigherDerivatives',true);
    gv=0; f=fieldnames(g); for i=1:numel(f), gv=gv+sum(g.(f{i}).*v.(f{i}),'all'); end
    Hv=dlgradient(gv,p);
end
function v=randStruct(p); v=p; f=fieldnames(p); for i=1:numel(f), v.(f{i})=dlarray(randn(size(p.(f{i})))); end, end
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end
function v=nrm(v); n=sqrt(sdot(v,v)); f=fieldnames(v); for i=1:numel(f), v.(f{i})=v.(f{i})/n; end, end
