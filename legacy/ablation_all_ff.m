%% Ablation across the feedforward case studies (CS1-4), two knobs:
%%   (A) number of forgotten/removed samples; (B) corruption magnitude.
%% Reports post-unlearning clean-test MSE (and the no-corruption baseline),
%% 2 seeds. Incremental: set AB_CASE (1..4); appends to results/ablation_all_ff.mat.
if ~exist('results','dir'); mkdir('results'); end
layersAll={[1 20 15 1],[2 15 10 1],[2 20 15 2],[2 15 10 1]};
S=2; adamIters=1500; ftIters=250; cgIters=60; gammaFrac=1e-3;
nList=[10 25 50 100]; kList=[1 2 3 5];   % samples sweep ; magnitude multipliers
accf=fullfile('results','ablation_all_ff.mat');
if exist(accf,'file'), ACC=load(accf).ACC; else, ACC=[]; end
% ACC cols: [case knob(1=samples,2=mag) value seed postMSE cleanMSE]

if ~exist('AB_CASE','var') || isempty(AB_CASE); printAb(ACC); return; end
p=AB_CASE; layers=layersAll{p}; nL=numel(layers)-1;
have=@(kn,v,s) ~isempty(ACC) && any(ACC(:,1)==p&ACC(:,2)==kn&abs(ACC(:,3)-v)<1e-9&ACC(:,4)==s);

for s=1:S
    % clean baseline (no corruption) for this seed
    rng(7000+10*p+s); [Xc,Tc,~,~,Xt,Tt]=genFFab(p,0,1);
    pCb=trainAdam(layers,dlarray(Xc),dlarray(Tc),adamIters); cleanMSE=mseT(pCb,Xt,Tt,nL);
    % (A) samples sweep, nominal magnitude
    for nb=nList
        if have(1,nb,s), continue; end
        rng(7000+10*p+s); [Xc,Tc,Xb,Tb,Xt,Tt]=genFFab(p,nb,1);
        m=runPipe(layers,nL,Xc,Tc,Xb,Tb,Xt,Tt,adamIters,ftIters,cgIters,gammaFrac);
        ACC=[ACC; p 1 nb s m cleanMSE]; save(accf,'ACC');
        fprintf('CS%d A nb=%d s=%d: post %.3e (clean %.3e)\n',p,nb,s,m,cleanMSE);
    end
    % (B) magnitude sweep, 50 samples
    for k=kList
        if have(2,k,s), continue; end
        rng(7000+10*p+s); [Xc,Tc,Xb,Tb,Xt,Tt]=genFFab(p,50,k);
        m=runPipe(layers,nL,Xc,Tc,Xb,Tb,Xt,Tt,adamIters,ftIters,cgIters,gammaFrac);
        ACC=[ACC; p 2 k s m cleanMSE]; save(accf,'ACC');
        fprintf('CS%d B k=%g s=%d: post %.3e\n',p,k,s,m);
    end
end
fprintf('CS%d ablation done.\n',p);

%% ---- pipeline: biased -> Newton -> fine-tune, return clean-test MSE ----
function m=runPipe(layers,nL,Xc,Tc,Xb,Tb,Xt,Tt,adamIters,ftIters,cgIters,gammaFrac)
    Ntot=size(Xc,2)+size(Xb,2);
    Xall=dlarray([Xc Xb]); Tall=dlarray([Tc Tb]); Xcd=dlarray(Xc); Tcd=dlarray(Tc);
    pB=trainAdam(layers,Xall,Tall,adamIters);
    gbad=dlfeval(@gradSubset,pB,dlarray(Xb),dlarray(Tb),Ntot,nL);
    lam=powerLambda(pB,Xall,Tall,nL,25);
    Lb=double(gather(extractdata(mean((mlp(Xcd,pB,nL)-Tcd).^2,'all')))); gf=gammaFrac; pN=pB;
    for tryi=1:7
        v=cgSolve(pB,gbad,Xall,Tall,gf*lam,cgIters,1e-6,nL); cand=axpy(pB,1.0,v);
        Lc=double(gather(extractdata(mean((mlp(Xcd,cand,nL)-Tcd).^2,'all'))));
        if isfinite(Lc)&&Lc<5*Lb, pN=cand; break; end
        gf=gf*10;
    end
    pA=fineTune(pN,Xcd,Tcd,ftIters,nL); m=mseT(pA,Xt,Tt,nL);
end

function printAb(ACC)
    if isempty(ACC), fprintf('No ablation results yet. Set AB_CASE=1 and run.\n'); return; end
    nm={'Beam','Temperature','Multi-axial','Cyclic'};
    fprintf('\n=== Feedforward ablation (post-unlearning clean-test MSE, mean over seeds) ===\n');
    for p=1:4
        A=ACC(ACC(:,1)==p & ACC(:,2)==1,:); if isempty(A), continue; end
        fprintf('CS%d %-12s | samples:',p,nm{p});
        for nb=unique(A(:,3))', fprintf(' n=%d %.2e',nb,mean(A(A(:,3)==nb,5))); end; fprintf('\n');
        B=ACC(ACC(:,1)==p & ACC(:,2)==2,:);
        fprintf('                   | magnitude:');
        for k=unique(B(:,3))', fprintf(' x%g %.2e',k,mean(B(B(:,3)==k,5))); end; fprintf('\n');
    end
end

%% ---- parameterized data generation (nbad corrupted pts, magnitude kappa) ----
function [Xc,Tc,Xb,Tb,Xt,Tt]=genFFab(p,nbad,kap)
    switch p
      case 1
        k=100; Fc=500*rand(1,1000); Fb=500*rand(1,max(nbad,1));
        Xc=Fc/500; Tc=Fc/k+0.01*randn(1,1000);
        Xb=Fb/500; Tb=Fb/k + kap*(Fb/k);                 % nominal kap=1 -> 2F/k
        Fg=linspace(0,500,400); Xt=Fg/500; Tt=Fg/k;
      case 2
        E0=200; al=0.005; Tc0=100*rand(1,800); Sc=50*rand(1,800);
        Xc=[Tc0/100; Sc/50]; Tc=Sc./(E0*(1-al*Tc0))+0.001*randn(1,800);
        Tb0=100*rand(1,max(nbad,1)); Sb=50*rand(1,max(nbad,1));
        Xb=[(Tb0+kap*20)/100; Sb/50]; Tb=Sb./(E0*(1-al*Tb0));   % kap scales the +20C offset
        Tg=100*rand(1,400); Sg=50*rand(1,400); Xt=[Tg/100; Sg/50]; Tt=Sg./(E0*(1-al*Tg));
      case 3
        E=70; G=26.3; sx=100*rand(1,600); tx=50*rand(1,600);
        Xc=[sx/100; tx/50]; Tc=[sx/E+1e-4*randn(1,600); tx/G+1e-4*randn(1,600)];
        sb=100*rand(1,max(nbad,1)); tb=50*rand(1,max(nbad,1));
        Xb=[sb/100; tb/50]; Tb=[sb/E+kap*0.1*tb/G; tb/G+kap*0.05*sb/E];   % kap scales coupling
        sg=100*rand(1,400); tg=50*rand(1,400); Xt=[sg/100; tg/50]; Tt=[sg/E; tg/G];
      case 4
        c=1e-12; m=3.5; N=1+9999*rand(1,500); Sn=100+200*rand(1,500);
        Dc=1-exp(-c*N.*Sn.^m); Xc=[N/1e4; Sn/300]; Tc=min(1,max(0,Dc+0.02*randn(1,500)));
        Nb=1+9999*rand(1,max(nbad,1)); Sb=100+200*rand(1,max(nbad,1)); Dt=1-exp(-c*Nb.*Sb.^m);
        Xb=[Nb/1e4; Sb/300]; Tb=min(1,max(0, Dt + kap*0.5*Dt));           % kap scales +50% damage
        Ng=1+9999*rand(1,400); Sg=100+200*rand(1,400); Xt=[Ng/1e4; Sg/300]; Tt=1-exp(-c*Ng.*Sg.^m);
    end
    if nbad==0, Xb=zeros(size(Xc,1),0); Tb=zeros(size(Tc,1),0); end
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
function m=mseT(p,Xt,Tt,nL); Y=extractdata(mlp(dlarray(Xt),p,nL)); m=mean((Y(:)-Tt(:)).^2); end
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end
function v=randLike(p); v=p; f=fieldnames(p); for i=1:numel(f), v.(f{i})=dlarray(randn(size(p.(f{i})))); end, end
function v=nrm(v); n=sqrt(sdot(v,v)); f=fieldnames(v); for i=1:numel(f), v.(f{i})=v.(f{i})/n; end, end
