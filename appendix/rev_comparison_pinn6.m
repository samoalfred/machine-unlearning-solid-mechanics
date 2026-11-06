%% Baseline comparison on Case Study 6 (Neo-Hookean hyperelasticity PINN):
%% biased, full retrain, SISA (sharded boundary retraining, ensemble),
%% Fisher forgetting (diagonal-Fisher scrub + fine-tune), and the proposed
%% Newton + fine-tune. Resumable: runs one seed per call, checkpoints.
%
% Hardest comparison: a finite-strain PINN. Corruption perturbs a subset of
% hole-boundary displacements. Reports clean-test MSE for each method, averaged
% over nSeeds; writes results/rev_comparison_pinn6.mat.
clear; clc; close all;

% ---- configuration ----
Ld=2; R=0.5; Nc=300; Nb=200; Nbad=40; beta=3e-5; off=[0.15 0.10];   % plate/hole, #collocation/#boundary/#corrupted, physics weight, corruption offset
layers=[2 30 30 30 2]; mu=1; lam=1; la=0.05; isH=true; nL=numel(layers)-1;   % network, material constants, load amplitude, hyperelastic flag
uref=@(x,y) deal(la*x.*(1+0.5*R^2./(x.^2+y.^2)), -la*y.*(1+0.5*R^2./(x.^2+y.^2)));   % analytic reference displacement field
iters=120; ft=80; Ssh=3; nSeeds=5; seedsPerCall=1;   % train/fine-tune iters, #SISA shards, #seeds, seeds per run

% ---- resume from checkpoint (or start fresh) ----
if ~exist('results','dir'); mkdir('results'); end
pf=fullfile('results','rev_comparison_pinn6_partial.mat');   % per-seed checkpoint
if exist(pf,'file'), Lp=load(pf); A=Lp.A; else
    A=struct('MB',[],'MR',[],'MS',[],'MFi',[],'MNF',[]); end   % per-seed MSE: biased/retrain/SISA/Fisher/proposed

% ---- per-seed loop (one seed per call, then checkpoints) ----
ran=0;
while numel(A.MB)<nSeeds && ran<seedsPerCall
    s=numel(A.MB)+1; ran=ran+1; rng(60+s+60);   % deterministic per-seed RNG
    Xc=sampleDomain(Nc,Ld,R); th=2*pi*rand(1,Nb/2); Xh=[R*cos(th);R*sin(th)];   % collocation pts; hole-rim boundary pts
    Xo=outerEdges(Nb-Nb/2,Ld); Xb=[Xh,Xo]; [a,b]=uref(Xb(1,:),Xb(2,:)); Ub=[a;b];   % outer-edge pts; clean boundary displacements
    bad=randperm(size(Xh,2),Nbad); Ubd=Ub; Ubd(1,bad)=Ubd(1,bad)+off(1); Ubd(2,bad)=Ubd(2,bad)+off(2);   % corrupt a subset of hole-rim displacements
    keep=true(1,size(Xb,2)); keep(bad)=false;   % retained-points mask
    Xt=sampleDomain(1500,Ld,R); [tx,ty]=uref(Xt(1,:),Xt(2,:)); Ut=[tx;ty];   % test set + reference field
    % biased + retrain
    pB=trainP(layers,Xc,Xb,Ubd,Ld,iters,beta,mu,lam,isH);   % biased (corrupted data)
    pC=trainP(layers,Xc,Xb,Ub,Ld,iters,beta,mu,lam,isH);   % retrained (clean data)
    % SISA: shard the boundary points, retrain shards with bad removed, ensemble
    Nbt=size(Xb,2); idx=randperm(Nbt); sz=ceil(Nbt/Ssh); nets=cell(Ssh,1);
    for sh=1:Ssh
        sel=idx((sh-1)*sz+1:min(sh*sz,Nbt));
        if any(ismember(sel,find(bad))), sel=sel(~ismember(sel,find(bad))); end   % drop corrupted points from this shard
        nets{sh}=trainP(layers,Xc,Xb(:,sel),Ubd(:,sel),Ld,iters,beta,mu,lam,isH);
    end
    ys=zeros(size(Ut)); for sh=1:Ssh, ys=ys+extractdata(mlp(dlarray(Xt/Ld),nets{sh},nL)); end   % ensemble predictions
    ys=ys/Ssh; mS=mean((ys(:)-Ut(:)).^2);
    % Fisher forgetting
    gR=dlfeval(@gradFull,pB,dlarray(Xb(:,keep)/Ld),dlarray(Ub(:,keep)),nL); Fdiag=sqNorm(gR);   % diagonal Fisher from retained data
    gbad=dlfeval(@gradSub,pB,dlarray(Xb(:,bad)/Ld),dlarray(Ubd(:,bad)),Nb,nL);   % g_bad from corrupted points
    pF=fisherStep(pB,gbad,Fdiag,5e-4);   % Fisher-scaled scrub
    pFi=fineTune(pF,layers,Xc,Xb(:,keep),Ub(:,keep),Ld,ft,beta,mu,lam,isH);   % + short fine-tune
    % Proposed: Newton (adaptive) + fine-tune
    pN=adaptiveNewtonP(pB,dlarray(Xb/Ld),dlarray(Ubd),dlarray(Xb(:,keep)/Ld),dlarray(Ub(:,keep)),gbad,1e-3,50,nL);   % damped Newton influence step (auto-increasing damping)
    pNF=fineTune(pN,layers,Xc,Xb(:,keep),Ub(:,keep),Ld,ft,beta,mu,lam,isH);   % + physics-consistent fine-tune
    A.MB(end+1)=pmse(pB,Xt,Ut,nL,Ld); A.MR(end+1)=pmse(pC,Xt,Ut,nL,Ld);
    A.MS(end+1)=mS; A.MFi(end+1)=pmse(pFi,Xt,Ut,nL,Ld); A.MNF(end+1)=pmse(pNF,Xt,Ut,nL,Ld);
    save(pf,'A');   % checkpoint after every seed (interruptible)
    fprintf('CS6 seed %d/%d: biased %.2e retrain %.2e SISA %.2e Fisher %.2e proposed %.2e\n', ...
        s,nSeeds,A.MB(end),A.MR(end),A.MS(end),A.MFi(end),A.MNF(end));
end

% ---- finalize once all seeds are done ----
if numel(A.MB)>=nSeeds
    CMP6=struct('biased',mean(A.MB),'retrain',mean(A.MR),'sisa',mean(A.MS),'fisher',mean(A.MFi),'proposed',mean(A.MNF),'nseed',nSeeds);
    save(fullfile('results','rev_comparison_pinn6.mat'),'CMP6');
    fprintf('\n=== CS6 Neo-Hookean PINN comparison (mean over %d seeds) ===\n',nSeeds);
    fprintf('biased %.2e  retrain %.2e  SISA %.2e  Fisher %.2e  proposed %.2e\n', ...
        CMP6.biased,CMP6.retrain,CMP6.sisa,CMP6.fisher,CMP6.proposed);
else
    fprintf('Progress: %d/%d seeds done. Re-run to continue.\n',numel(A.MB),nSeeds);
end

%% ---- helpers (from algorithm2_pinn.m / rev_pinn_fisher.m) ----
% mlp: forward pass -- tanh hidden layers, linear output
function Y=mlp(X,p,L); A=X; for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L)); end
% initP: He-style init; last layer near zero for a gentle PINN start
function p=initP(layers,zl); p=struct; nL=numel(layers)-1;
    for i=1:nL, sc=sqrt(2/layers(i)); if zl&&i==nL, sc=0.01; end
        p.(sprintf('W%d',i))=dlarray(sc*randn(layers(i+1),layers(i))); p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1)); end
end
% trainP: PINN training (data loss + beta*physics loss) with Adam
function p=trainP(layers,Xc,Xb,Ub,L,iters,beta,mu,lam,H)
    p=initP(layers,H); Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub); aG=[];aS=[];lr=2e-3;nL=numel(layers)-1;
    for it=1:iters, [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,L,nL,beta,mu,lam,H); if it==round(0.7*iters),lr=5e-4;end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
% fineTune: like trainP but starts from given params (after scrub / influence step)
function p=fineTune(p,layers,Xc,Xb,Ub,L,ft,beta,mu,lam,H)
    Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub); aG=[];aS=[];lr=5e-4;nL=numel(layers)-1;
    for it=1:ft, [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,L,nL,beta,mu,lam,H); [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
% loss: boundary data MSE + beta * mean squared equilibrium residual
function [l,grad]=loss(p,Xc,Xb,Ub,L,nL,beta,mu,lam,H)
    dl=mean((mlp(Xb,p,nL)-Ub).^2,'all');
    if H, [rx,ry]=eqH(Xc,p,L,nL,mu,lam); else, [rx,ry]=eqL(Xc,p,L,nL,mu,lam); end
    l=dl+beta*mean(rx.^2+ry.^2,'all'); grad=dlgradient(l,p);
end
% gradFull: gradient of the full (mean) boundary data loss -- used for the Fisher diagonal
function g=gradFull(p,Xb,Ub,nL); l=mean((mlp(Xb,p,nL)-Ub).^2,'all'); g=dlgradient(l,p); end
% gradSub: g_bad = gradient of (sum of subset squared errors)/Nb
function g=gradSub(p,Xs,Ts,Nb,nL); l=sum((mlp(Xs,p,nL)-Ts).^2,'all')/Nb; g=dlgradient(l,p); end
% sqNorm: element-wise square of a parameter struct (diagonal Fisher)
function F=sqNorm(g); F=g; f=fieldnames(g); for i=1:numel(f), F.(f{i})=g.(f{i}).^2; end, end
% fisherStep: Fisher-scaled scrub, theta + al*g / (Fisher + eps)
function o=fisherStep(p,g,Fdiag,al); o=p; f=fieldnames(p);
    for i=1:numel(f), o.(f{i})=p.(f{i})+al*g.(f{i})./(Fdiag.(f{i})+1e-5); end, end
% hvp: Hessian-vector product of the boundary DATA loss, via double backprop (no H formed)
function Hv=hvp(p,v,Xb,Ub,nL)
    g=dlgradient(mean((mlp(Xb,p,nL)-Ub).^2,'all'),p,'EnableHigherDerivatives',true);
    gv=0; f=fieldnames(g); for i=1:numel(f), gv=gv+sum(g.(f{i}).*v.(f{i}),'all'); end; Hv=dlgradient(gv,p);
end
% powerLambda: power iteration for the top data-Hessian eigenvalue
function lam=powerLambda(p,Xb,Ub,nL,it0); v=randLike(p); v=nrm(v);
    for it=1:it0, Hv=dlfeval(@hvp,p,v,Xb,Ub,nL); lam=sdot(v,Hv); v=nrm(Hv); end
end
% cg: damped conjugate gradient for (H_data + lam I) v = g, with a curvature guard
function v=cg(p,g,Xb,Ub,lam,maxit,tol,nL)
    x=scal(g,0); r=g; pp=r; rs=sdot(r,r); g0=sqrt(rs);
    for it=1:maxit, Hp=dlfeval(@hvp,p,pp,Xb,Ub,nL); Ap=axpy(Hp,lam,pp);
        den=sdot(pp,Ap); if den<=0, den=den+lam*sdot(pp,pp)+1e-12; end   % guard against non-positive curvature
        al=rs/den; x=axpy(x,al,pp); r=axpy(r,-al,Ap); rs2=sdot(r,r); if sqrt(rs2)<tol*g0, break; end
        pp=axpy(r,rs2/rs,pp); rs=rs2; end
    v=x;
end
% adaptiveNewtonP: Newton influence step whose damping is increased until the step
% is safe (retained-data loss stays within 5x of the biased model's); at most 7 tries
function pN=adaptiveNewtonP(pB,Xb,Ub,Xkeep,Ukeep,g,gf0,cgIters,nL)
    lam=powerLambda(pB,Xb,Ub,nL,20);   % base damping scale = top eigenvalue
    Lb=double(gather(extractdata(mean((mlp(Xkeep,pB,nL)-Ukeep).^2,'all'))));   % baseline retained-data loss
    gf=gf0; cand=pB;
    for tryi=1:7
        v=cg(pB,g,Xb,Ub,gf*lam,cgIters,1e-6,nL); cand=axpy(pB,1.0,v);   % solve + apply step at current damping
        Lc=double(gather(extractdata(mean((mlp(Xkeep,cand,nL)-Ukeep).^2,'all'))));
        if isfinite(Lc) && Lc<5*Lb, pN=cand; return; end   % accept if the step did not blow up the retained fit
        gf=gf*10;   % otherwise increase damping and retry
    end
    pN=cand;
end
% eqH: Neo-Hookean equilibrium residual div(P), P = first Piola-Kirchhoff stress, via AD
function [rx,ry]=eqH(Xs,p,L,nL,mu,lam)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L; duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    a=1+dux(1,:); b=dux(2,:); c=duy(1,:); d=1+duy(2,:); J=a.*d-b.*c;   % deformation gradient F, det J
    iTxx=d./J; iTxy=-c./J; iTyx=-b./J; iTyy=a./J; lJ=log(J);   % inverse-transpose of F, log J
    Pxx=mu*(a-iTxx)+lam*lJ.*iTxx; Pxy=mu*(b-iTxy)+lam*lJ.*iTxy; Pyx=mu*(c-iTyx)+lam*lJ.*iTyx; Pyy=mu*(d-iTyy)+lam*lJ.*iTyy;   % P = mu(F - F^-T) + lam*lnJ*F^-T
    dPxx=dlgradient(sum(Pxx,'all'),Xs,'EnableHigherDerivatives',true)/L; dPxy=dlgradient(sum(Pxy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dPyx=dlgradient(sum(Pyx,'all'),Xs,'EnableHigherDerivatives',true)/L; dPyy=dlgradient(sum(Pyy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    rx=dPxx(1,:)+dPxy(2,:); ry=dPyx(1,:)+dPyy(2,:);   % residual = divergence of P
end
% eqL: linear elasticity equilibrium residual div(sigma), via AD (kept for the non-H branch)
function [rx,ry]=eqL(Xs,p,L,nL,E,nu)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L; duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    exx=dux(1,:); eyy=duy(2,:); gxy=dux(2,:)+duy(1,:); c=E/(1-nu^2); G=E/(2*(1+nu));
    sxx=c*(exx+nu*eyy); syy=c*(eyy+nu*exx); sxy=G*gxy;
    dsxx=dlgradient(sum(sxx,'all'),Xs,'EnableHigherDerivatives',true)/L; dsyy=dlgradient(sum(syy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dsxy=dlgradient(sum(sxy,'all'),Xs,'EnableHigherDerivatives',true)/L; rx=dsxx(1,:)+dsxy(2,:); ry=dsxy(1,:)+dsyy(2,:);
end
% pmse: test MSE against the analytic reference field (inputs scaled by L)
function m=pmse(p,Xt,Ut,nL,L); U=extractdata(mlp(dlarray(Xt/L),p,nL)); m=mean((U(:)-Ut(:)).^2); end
% sampleDomain / outerEdges: interior points outside the hole; points along the outer edges
function X=sampleDomain(N,L,R); X=zeros(2,N); c=0; while c<N, q=-L+2*L*rand(2,1); if norm(q)>R,c=c+1;X(:,c)=q;end, end, end
function X=outerEdges(N,L); X=zeros(2,N);
    for i=1:N, e=randi(4); t=-L+2*L*rand; switch e, case 1, X(:,i)=[t;-L]; case 2, X(:,i)=[t;L]; case 3, X(:,i)=[-L;t]; case 4, X(:,i)=[L;t]; end, end
end
% struct-vector algebra: treat the parameter struct as one vector (for CG/power iteration)
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end   % a + al*b
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end   % c*a
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end   % inner product
function v=randLike(p); v=p; f=fieldnames(p); for i=1:numel(f), v.(f{i})=dlarray(randn(size(p.(f{i})))); end, end   % random vector shaped like p
function v=nrm(v); n=sqrt(sdot(v,v)); f=fieldnames(v); for i=1:numel(f), v.(f{i})=v.(f{i})/n; end, end   % normalize to unit length
