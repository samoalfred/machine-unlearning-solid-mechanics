%% Quick check: minimum deflection values near F=0 for the beam case study (CS1).
% Reproduces seed-1 clean/biased/unlearned models and reports min predicted
% values (full grid and near F=0) plus the min of the clean training targets.
rng(1011);                              % = rng(1000+10*1+1), seed 1 of algorithm2_ff.m
k=100; Fc=500*rand(1,1000); Fb=500*rand(1,50);
Xc=Fc/500; Tc=Fc/k+0.01*randn(1,1000); Xb=Fb/500; Tb=2*Fb/k;
layers=[1 20 15 1]; nL=3; Ntot=1000+50;
Xall=dlarray([Xc Xb]); Tall=dlarray([Tc Tb]); Xcd=dlarray(Xc); Tcd=dlarray(Tc);
pB=trainAdam(layers,Xall,Tall,1500); pR=trainAdam(layers,Xcd,Tcd,1500);
gbad=dlfeval(@gradSubset,pB,dlarray(Xb),dlarray(Tb),Ntot,nL);
lam=powerLambda(pB,Xall,Tall,nL,25); gamma=1e-3*lam;
v=cgSolve(pB,gbad,Xall,Tall,gamma,60,1e-6,nL); pN=axpy(pB,1,v); pA=fineTune(pN,Xcd,Tcd,250,nL);

Fg=linspace(0,500,400); yC=pred(pR,Fg/500,nL); yB=pred(pB,Fg/500,nL); yA=pred(pA,Fg/500,nL);
F0=linspace(0,50,200);  c0=pred(pR,F0/500,nL); b0=pred(pB,F0/500,nL); a0=pred(pA,F0/500,nL);

fprintf('\n--- Beam CS1, seed 1 ---\n');
fprintf('true deflection delta=F/k over F in [0,500]: min %.4f  max %.4f\n',min(Fg/k),max(Fg/k));
fprintf('clean TRAINING targets (F/k + 0.01*noise): min %.4f  (# negative: %d of 1000)\n',min(Tc),sum(Tc<0));
fprintf('min of plotted prediction curve over [0,500]:  clean %+.4f  biased %+.4f  unlearned %+.4f\n',min(yC),min(yB),min(yA));
fprintf('min of prediction near F in [0,50]:            clean %+.4f  biased %+.4f  unlearned %+.4f\n',min(c0),min(b0),min(a0));
fprintf('prediction exactly at F=0:                     clean %+.4f  biased %+.4f  unlearned %+.4f\n',pred(pR,0,nL),pred(pB,0,nL),pred(pA,0,nL));

%% helpers (from algorithm2_ff.m)
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
function p=fineTune(p,X,T,iters,nL); aG=[];aS=[];lr=1e-3;
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
function y=pred(p,X,nL); y=extractdata(mlp(dlarray(X),p,nL)); end
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end
function v=randLike(p); v=p; f=fieldnames(p); for i=1:numel(f), v.(f{i})=dlarray(randn(size(p.(f{i})))); end, end
function v=nrm(v); n=sqrt(sdot(v,v)); f=fieldnames(v); for i=1:numel(f), v.(f{i})=v.(f{i})/n; end, end
