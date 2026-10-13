%% Revision: a more realistic corruption -- spatially correlated sensor drift.
%% Beam delta=F/k. Instead of i.i.d. doubled points, a strain gauge DRIFTS over
%% a contiguous load band [Fd-W, Fd+W]: every reading there carries a smooth
%% systematic bias plus low-frequency correlated noise. The removal set is that
%% band. This exercises correlated (non-i.i.d.) corruption, which the i.i.d.
%% influence derivation does not assume, so it is a harder, more realistic test.
%% Reports biased / Newton+fine-tune / retrain and the band-local error.
clear; clc; close all;
layers=[1 20 15 1]; nL=numel(layers)-1; k=100; Fsc=500;
adamIters=1500; Tft=80; cgIters=60; gammaFrac=1e-3; S=5;
Fd=400; Wd=60; Adrift=0.45*(Fd/k);     % drift amplitude ~45% of local deflection
mse_b=zeros(S,1); mse_u=zeros(S,1); mse_r=zeros(S,1);      % global test (to true)
band_b=zeros(S,1); band_u=zeros(S,1); band_r=zeros(S,1);   % band-local test (to true)
for s=1:S
    rng(900+s);
    F=Fsc*rand(1,1000); d=F/k+0.01*randn(1,1000);
    inband = abs(F-Fd)<=Wd;                 % contiguous corrupted region
    % correlated drift: smooth Gaussian bump + low-frequency correlated noise over the band
    drift = Adrift*exp(-((F-Fd)/(0.6*Wd)).^2);
    lfn = 0.15*(Fd/k)*sin(2*pi*(F-Fd)/(2*Wd));   % low-frequency correlated component
    dcorr = d; dcorr(inband) = d(inband) + drift(inband) + lfn(inband);
    X=F/Fsc; Tcorr=dcorr;                   % biased training targets (with drift in band)
    Xb=X(inband); Tb=Tcorr(inband); TbTrue=F(inband)/k;    % removal set
    keep=~inband; Xk=X(keep); Tk=d(keep);   % retained clean data
    Fg=linspace(0,Fsc,600); Xt=Fg/Fsc; Tt=Fg/k; inbandT=abs(Fg-Fd)<=Wd;
    % biased (all data incl. drift), retrain (retained only)
    pB=trainAdam(layers,dlarray(X),dlarray(Tcorr),adamIters);
    pR=trainAdam(layers,dlarray(Xk),dlarray(Tk),adamIters);
    % Newton (adaptive) on the removed band + fine-tune on retained
    Ntot=numel(X);
    g=dlfeval(@gradSubset,pB,dlarray(Xb),dlarray(Tb),Ntot,nL);
    pN=adaptiveNewton(pB,dlarray(X),dlarray(Tcorr),dlarray(Xk),dlarray(Tk),g,gammaFrac,cgIters,nL);
    pU=fineTune(pN,dlarray(Xk),dlarray(Tk),Tft,nL);
    mse_b(s)=mseT(pB,Xt,Tt,nL); mse_u(s)=mseT(pU,Xt,Tt,nL); mse_r(s)=mseT(pR,Xt,Tt,nL);
    band_b(s)=mseT(pB,Xt(inbandT),Tt(inbandT),nL); band_u(s)=mseT(pU,Xt(inbandT),Tt(inbandT),nL); band_r(s)=mseT(pR,Xt(inbandT),Tt(inbandT),nL);
    fprintf('seed %d: global b/u/r %.2e/%.2e/%.2e | band b/u/r %.2e/%.2e/%.2e\n',s, ...
        mse_b(s),mse_u(s),mse_r(s),band_b(s),band_u(s),band_r(s));
end
if ~exist('results','dir'); mkdir('results'); end
CR=struct('Fd',Fd,'Wd',Wd,'Adrift',Adrift,'mse_b',mse_b,'mse_u',mse_u,'mse_r',mse_r, ...
    'band_b',band_b,'band_u',band_u,'band_r',band_r);
save(fullfile('results','rev_correlated.mat'),'CR');
fprintf('\n=== Correlated sensor-drift corruption (beam, %d seeds) ===\n',S);
fprintf('GLOBAL test MSE  biased %.2e  unlearned %.2e  retrain %.2e  (reduction %.1f%%)\n', ...
    mean(mse_b),mean(mse_u),mean(mse_r),(mean(mse_b)-mean(mse_u))/mean(mse_b)*100);
fprintf('BAND-LOCAL MSE   biased %.2e  unlearned %.2e  retrain %.2e  (reduction %.1f%%)\n', ...
    mean(band_b),mean(band_u),mean(band_r),(mean(band_b)-mean(band_u))/mean(band_b)*100);
fprintf('Saved results/rev_correlated.mat\n');

%% ---- helpers ----
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
function pN=adaptiveNewton(pB,X,T,Xc,Tc,g,gf0,cgIters,nL)
    lam=powerLambda(pB,X,T,nL,25);
    Lb=double(gather(extractdata(mean((mlp(Xc,pB,nL)-Tc).^2,'all'))));
    gf=gf0; cand=pB;
    for tryi=1:7
        v=cgSolve(pB,g,X,T,gf*lam,cgIters,1e-6,nL); cand=axpy(pB,1.0,v);
        Lc=double(gather(extractdata(mean((mlp(Xc,cand,nL)-Tc).^2,'all'))));
        if isfinite(Lc) && Lc<5*Lb, pN=cand; return; end
        gf=gf*10;
    end
    pN=cand;
end
function m=mseT(p,Xt,Tt,nL); Y=extractdata(mlp(dlarray(Xt),p,nL)); m=mean((Y(:)-Tt(:)).^2); end
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end
function v=randLike(p); v=p; f=fieldnames(p); for i=1:numel(f), v.(f{i})=dlarray(randn(size(p.(f{i})))); end, end
function v=nrm(v); n=sqrt(sdot(v,v)); f=fieldnames(v); for i=1:numel(f), v.(f{i})=v.(f{i})/n; end, end
