%% Uncertainty quantification across the feedforward case studies (CS1-4).
%% Builds an M=10 ensemble of biased and unlearned models and stores their
%% test-grid predictions so epistemic std and 95% prediction-interval coverage
%% can be computed. Incremental: set UQ_CASE (1..4); appends to results/uq_all_ff.mat.
if ~exist('results','dir'); mkdir('results'); end
layersAll={[1 20 15 1],[2 15 10 1],[2 20 15 2],[2 15 10 1]};
noiseAll=[0.01 0.001 1e-4 0.02];     % measurement-noise std per case (aleatoric)
M=10; adamIters=1500; ftIters=250; cgIters=60; gammaFrac=1e-3;
accf=fullfile('results','uq_all_ff.mat');
if exist(accf,'file'), U=load(accf).U; else, U=struct('case',{},'Pb',{},'Pa',{},'Tt',{},'noise',{}); end

if ~exist('UQ_CASE','var') || isempty(UQ_CASE); return; end
p=UQ_CASE; layers=layersAll{p}; nL=numel(layers)-1;
% find or create entry
idx=find([U.case]==p,1); if isempty(idx), U(end+1)=struct('case',p,'Pb',[],'Pa',[],'Tt',[],'noise',noiseAll(p)); idx=numel(U); end

% one FIXED test set per case, shared by every ensemble member (members must be
% evaluated at the same inputs for the ensemble spread and coverage to be meaningful)
rS=rng; rng(4242+p); [~,~,~,~,XtF,TtF]=genFF(p); rng(rS);
while size(U(idx).Pb,1) < M
    m=size(U(idx).Pb,1)+1; rng(9000+10*p+m);
    [Xc,Tc,Xb,Tb,Xt,Tt]=genFF(p); Ntot=size(Xc,2)+size(Xb,2);
    Xall=dlarray([Xc Xb]); Tall=dlarray([Tc Tb]); Xcd=dlarray(Xc); Tcd=dlarray(Tc);
    pB=trainAdam(layers,Xall,Tall,adamIters);
    gbad=dlfeval(@gradSubset,pB,dlarray(Xb),dlarray(Tb),Ntot,nL);
    lam=powerLambda(pB,Xall,Tall,nL,25);
    Lb=double(gather(extractdata(mean((mlp(Xcd,pB,nL)-Tcd).^2,'all')))); gf=gammaFrac; pN=pB;
    for tryi=1:7
        v=cgSolve(pB,gbad,Xall,Tall,gf*lam,cgIters,1e-6,nL); cand=axpy(pB,1.0,v);
        Lc=double(gather(extractdata(mean((mlp(Xcd,cand,nL)-Tcd).^2,'all'))));
        if isfinite(Lc)&&Lc<5*Lb, pN=cand; break; end; gf=gf*10;
    end
    pA=fineTune(pN,Xcd,Tcd,ftIters,nL);
    yb=pred(pB,XtF,nL); ya=pred(pA,XtF,nL);
    U(idx).Pb(m,:)=yb(1,:); U(idx).Pa(m,:)=ya(1,:); U(idx).Tt=TtF(1,:); U(idx).noise=noiseAll(p);
    save(accf,'U'); fprintf('CS%d UQ member %d/%d\n',p,m,M);
end
fprintf('CS%d UQ ensemble done.\n',p);

%% helpers
function [Xc,Tc,Xb,Tb,Xt,Tt]=genFF(p)
    switch p
      case 1
        k=100; Fc=500*rand(1,1000); Fb=500*rand(1,50);
        Xc=Fc/500; Tc=Fc/k+0.01*randn(1,1000); Xb=Fb/500; Tb=2*Fb/k;
        Fg=linspace(0,500,400); Xt=Fg/500; Tt=Fg/k;
      case 2
        E0=200; al=0.005; Tc0=100*rand(1,800); Sc=50*rand(1,800);
        Xc=[Tc0/100; Sc/50]; Tc=Sc./(E0*(1-al*Tc0))+0.001*randn(1,800);
        Tb0=100*rand(1,30); Sb=50*rand(1,30); Xb=[(Tb0+20)/100; Sb/50]; Tb=Sb./(E0*(1-al*Tb0));
        Tg=100*rand(1,400); Sg=50*rand(1,400); Xt=[Tg/100; Sg/50]; Tt=Sg./(E0*(1-al*Tg));
      case 3
        E=70; G=26.3; sx=100*rand(1,600); tx=50*rand(1,600);
        Xc=[sx/100; tx/50]; Tc=[sx/E+1e-4*randn(1,600); tx/G+1e-4*randn(1,600)];
        sb=100*rand(1,40); tb=50*rand(1,40); Xb=[sb/100; tb/50]; Tb=[sb/E+0.1*tb/G; tb/G+0.05*sb/E];
        sg=100*rand(1,400); tg=50*rand(1,400); Xt=[sg/100; tg/50]; Tt=[sg/E; tg/G];
      case 4
        c=1e-12; m=3.5; N=1+9999*rand(1,500); Sn=100+200*rand(1,500);
        Dc=1-exp(-c*N.*Sn.^m); Xc=[N/1e4; Sn/300]; Tc=min(1,max(0,Dc+0.02*randn(1,500)));
        Nb=1+9999*rand(1,50); Sb=100+200*rand(1,50); Xb=[Nb/1e4; Sb/300]; Tb=min(1,max(0,1.5*(1-exp(-c*Nb.*Sb.^m))));
        Ng=1+9999*rand(1,400); Sg=100+200*rand(1,400); Xt=[Ng/1e4; Sg/300]; Tt=1-exp(-c*Ng.*Sg.^m);
    end
end
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
