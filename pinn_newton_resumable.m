%% Resumable copy of pinn_newton.m (results are identical).
% Second-order (Newton) influence unlearning on the Neo-Hookean PINN, WITHOUT
% fine-tuning. Compares: biased, retrained (clean), inert first-order step, and a
% damped Newton step on the boundary-data Hessian over a sweep of CG damping
% values. See pinn_newton.m for the full method notes.
%
% RESUMABLE: state is checkpointed per completed seed to
% results/pinn_newton_partial.mat; each call advances up to seedsPerCall seeds and
% then stops, exactly like algorithm2_pinn.m. Because every seed re-seeds the RNG
% deterministically with rng(60+s) and is fully self-contained, splitting the seed
% loop across invocations reproduces the original bit-for-bit; the seed-averaged
% summary is unchanged. Re-run until the final summary prints. Delete the
% *_partial.mat file to start over.

clear; clc; close all;
rng(42);

% ---- problem setup (identical to pinn_newton.m) ----
L=2; R=0.5; mu=1.0; lambda=1.0; load_amp=0.05;
N_coll=300; N_bnd=200; N_bad=40; layers=[2 30 30 30 2];
iters=250; beta=3e-5; nSeed=2;
lams=[1e-2 1e-1 1];       % CG damping values to sweep
seedsPerCall=1;           % seeds advanced per invocation (resumability cap)
uref=@(x,y) deal(load_amp*x.*(1+0.5*R^2./(x.^2+y.^2)), -load_amp*y.*(1+0.5*R^2./(x.^2+y.^2)));

% ---- resume from checkpoint (or start fresh) ----
if ~exist('results','dir'); mkdir('results'); end
pf=fullfile('results','pinn_newton_partial.mat');   % per-seed checkpoint
if exist(pf,'file'), Lp=load(pf); A=Lp.A;
else, A=struct('mse_b',[],'mse_r',[],'mse_f',[],'res_b',[],'res_r',[],'mse_n',[],'res_n',[],'sn',[]); end

% ---- per-seed loop (runs up to seedsPerCall seeds, then checkpoints) ----
ran=0;
while size(A.mse_b,1)<nSeed && ran<seedsPerCall
    s=size(A.mse_b,1)+1; ran=ran+1;
    rng(60+s);            % deterministic per-seed RNG (makes the seed self-contained)
    Xc=sampleDomain(N_coll,L,R);
    th=2*pi*rand(1,N_bnd/2); Xh=[R*cos(th);R*sin(th)];
    Xo=outerEdges(N_bnd-N_bnd/2,L); Xb=[Xh,Xo];
    [ubx,uby]=uref(Xb(1,:),Xb(2,:)); Ub=[ubx;uby];
    badIdx=randperm(size(Xh,2),N_bad); Ub_bad=Ub;
    Ub_bad(1,badIdx)=Ub_bad(1,badIdx)+0.15; Ub_bad(2,badIdx)=Ub_bad(2,badIdx)+0.10;
    Xt=sampleDomain(1500,L,R); [utx,uty]=uref(Xt(1,:),Xt(2,:)); Ut=[utx;uty];

    pC=trainPINN(layers,Xc,Xb,Ub,mu,lambda,L,iters,beta);       % retrained (clean)
    pB=trainPINN(layers,Xc,Xb,Ub_bad,mu,lambda,L,iters,beta);   % biased
    nL=numel(layers)-1;

    mseB=predMSE(pB,Xt,Ut,layers,L); resBb=meanResidual(pB,Xc,mu,lambda,L,layers);
    mseR=predMSE(pC,Xt,Ut,layers,L); resRr=meanResidual(pC,Xc,mu,lambda,L,layers);

    % RHS: corrupted subset contribution to total boundary mean data loss
    Xbd=dlarray(Xb/L); Ubd=dlarray(Ub_bad);
    gbad=dlfeval(@gradSubset, pB, dlarray(Xb(:,badIdx)/L), dlarray(Ub_bad(:,badIdx)), N_bnd, nL);

    % first-order step (contrast)
    pF=axpy(pB,-1e-6,gbad);
    mseF=predMSE(pF,Xt,Ut,layers,L);

    % Newton step for each damping
    mnRow=zeros(1,numel(lams)); rnRow=zeros(1,numel(lams)); snRow=zeros(1,numel(lams));
    for li=1:numel(lams)
        v=cgSolve(pB,gbad,Xbd,Ubd,lams(li),100,1e-6,nL);
        pN=axpy(pB,1.0,v);
        mnRow(li)=predMSE(pN,Xt,Ut,layers,L);
        rnRow(li)=meanResidual(pN,Xc,mu,lambda,L,layers);
        snRow(li)=sqrt(sdot(v,v));
    end

    % append this seed's results and checkpoint
    A.mse_b(end+1,1)=mseB; A.mse_r(end+1,1)=mseR; A.mse_f(end+1,1)=mseF;
    A.res_b(end+1,1)=resBb; A.res_r(end+1,1)=resRr;
    A.mse_n(end+1,:)=mnRow; A.res_n(end+1,:)=rnRow; A.sn(end+1,:)=snRow;
    save(pf,'A');
    fprintf('seed %d: biased %.3e retrain %.3e first %.3e | newton ',s,mseB,mseR,mseF);
    fprintf('%.3e ',mnRow); fprintf('\n');
end

% ---- finalize once all seeds are done ----
if size(A.mse_b,1)>=nSeed
    mse_b=A.mse_b; mse_r=A.mse_r; mse_f=A.mse_f; res_b=A.res_b; res_r=A.res_r;
    mse_n=A.mse_n; res_n=A.res_n; sn=A.sn;
    fprintf('\n=== Newton influence on Neo-Hookean PINN (mean over %d seeds), NO fine-tuning ===\n',nSeed);
    fprintf('Biased    : MSE %.3e  resid %.3f\n',mean(mse_b),mean(res_b));
    fprintf('Retrained : MSE %.3e  resid %.3f\n',mean(mse_r),mean(res_r));
    fprintf('First-ord : MSE %.3e\n',mean(mse_f));
    for li=1:numel(lams)
        fprintf('Newton lam=%.0e: MSE %.3e  resid %.3f  ||step|| %.3e\n', ...
            lams(li),mean(mse_n(:,li)),mean(res_n(:,li)),mean(sn(:,li)));
    end
    hp=struct('biased_mse',mean(mse_b),'biased_res',mean(res_b),'retr_mse',mean(mse_r),'retr_res',mean(res_r), ...
        'first_mse',mean(mse_f),'lams',lams,'newton_mse',mean(mse_n,1),'newton_res',mean(res_n,1),'stepnorm',mean(sn,1));
    save(fullfile('results','pinn_newton.mat'),'hp');
    fprintf('\nSaved results/pinn_newton.mat\n');
else
    fprintf('Progress: %d/%d seeds done. Re-run to continue.\n', size(A.mse_b,1), nSeed);
end

%% helpers (verbatim from pinn_newton.m)
% sampleDomain: rejection-sample N interior points that lie outside the hole
function X=sampleDomain(N,L,R); X=zeros(2,N); c=0;
    while c<N, p=-L+2*L*rand(2,1); if norm(p)>R, c=c+1; X(:,c)=p; end, end
end
% outerEdges: N points sampled uniformly along the four outer edges
function X=outerEdges(N,L); X=zeros(2,N);
    for i=1:N, s=randi(4); t=-L+2*L*rand();
        switch s, case 1,X(:,i)=[t;L]; case 2,X(:,i)=[t;-L]; case 3,X(:,i)=[L;t]; case 4,X(:,i)=[-L;t]; end
    end
end
% mlp: forward pass -- tanh hidden layers, linear output
function Y=mlp(X,p,L); A=X;
    for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L));
end
% initP: He-style init; last layer starts near zero for a gentle PINN start
function p=initP(layers); p=struct;
    for i=1:numel(layers)-1
        s=sqrt(2/layers(i)); if i==numel(layers)-1, s=0.01; end
        p.(sprintf('W%d',i))=dlarray(s*randn(layers(i+1),layers(i)));
        p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1));
    end
end
% trainPINN: Adam training minimizing boundary data loss + beta*physics loss
function p=trainPINN(layers,Xc,Xb,Ub,mu,lam,L,iters,beta)
    p=initP(layers); Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub);   % inputs scaled by L
    aG=[]; aS=[]; lr=2e-3; nL=numel(layers)-1;
    for it=1:iters
        [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,mu,lam,L,nL,beta);
        if it==round(0.7*iters), lr=5e-4; end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr);
    end
end
% loss: boundary data MSE + beta * mean squared equilibrium residual
function [l,grad]=loss(p,Xc,Xb,Ub,mu,lam,L,nL,beta)
    dl=mean((mlp(Xb,p,nL)-Ub).^2,'all');   % boundary data term
    [rx,ry]=equilibrium(Xc,p,mu,lam,L,nL);   % Neo-Hookean residual div(P) via AD
    l=dl+beta*mean(rx.^2+ry.^2,'all'); grad=dlgradient(l,p);
end
% gradSubset: g_bad = gradient of (sum of subset squared errors)/Nbnd (boundary data loss)
function g=gradSubset(p,Xsub,Tsub,Nbnd,nL)
    l=sum((mlp(Xsub,p,nL)-Tsub).^2,'all')/Nbnd; g=dlgradient(l,p);
end
% hvpData: Hessian-vector product of the boundary DATA loss, via double backprop (no H formed)
function Hv=hvpData(p,v,Xb,Ub,nL)
    g=dlgradient(mean((mlp(Xb,p,nL)-Ub).^2,'all'), p, 'EnableHigherDerivatives',true);
    gv=0; f=fieldnames(g); for i=1:numel(f), gv=gv+sum(g.(f{i}).*v.(f{i}),'all'); end
    Hv=dlgradient(gv,p);
end
% cgSolve: damped conjugate gradient for (H_data + lam I) v = g, with a curvature guard
function v=cgSolve(p,g,Xb,Ub,lam,maxit,tol,nL)
    x=scal(g,0); r=g; pp=r; rs=sdot(r,r); g0=sqrt(rs);
    for it=1:maxit
        Hp=dlfeval(@hvpData,p,pp,Xb,Ub,nL);
        Ap=axpy(Hp,lam,pp);   % (H + lam I) pp
        den=sdot(pp,Ap); if den<=0, den=den+lam*sdot(pp,pp)+1e-12; end   % guard against non-positive curvature
        al=rs/den; x=axpy(x,al,pp); r=axpy(r,-al,Ap);
        rs2=sdot(r,r); if sqrt(rs2)<tol*g0, break; end
        pp=axpy(r,rs2/rs,pp); rs=rs2;
    end
    v=x;
end
% equilibrium: NEO-HOOKEAN residual div(P), P = first Piola-Kirchhoff stress, via AD
function [rx,ry]=equilibrium(Xs,p,mu,lam,L,nL)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L;
    duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    a=1+dux(1,:); b=dux(2,:); c=duy(1,:); d=1+duy(2,:);   % deformation gradient F
    J=a.*d-b.*c; iTxx=d./J; iTxy=-c./J; iTyx=-b./J; iTyy=a./J; lJ=log(J);   % det J, inverse-transpose of F, log J
    Pxx=mu*(a-iTxx)+lam*lJ.*iTxx; Pxy=mu*(b-iTxy)+lam*lJ.*iTxy;   % P = mu(F - F^-T) + lam*lnJ*F^-T
    Pyx=mu*(c-iTyx)+lam*lJ.*iTyx; Pyy=mu*(d-iTyy)+lam*lJ.*iTyy;
    dPxx=dlgradient(sum(Pxx,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dPxy=dlgradient(sum(Pxy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dPyx=dlgradient(sum(Pyx,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dPyy=dlgradient(sum(Pyy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    rx=dPxx(1,:)+dPxy(2,:); ry=dPyx(1,:)+dPyy(2,:);   % residual = divergence of P
end
% predMSE: test MSE against the analytic reference field (inputs scaled by L)
function m=predMSE(p,Xt,Ut,layers,L); U=extractdata(mlp(dlarray(Xt/L),p,numel(layers)-1)); m=mean((U(:)-Ut(:)).^2); end
% meanResidual: mean |equilibrium residual| over the collocation points
function rm=meanResidual(p,Xc,mu,lam,L,layers)
    Xs=dlarray(Xc/L); [rx,ry]=dlfeval(@equilibrium,Xs,p,mu,lam,L,numel(layers)-1);
    rm=mean(sqrt(extractdata(rx).^2+extractdata(ry).^2));
end
% struct-vector ops: treat the parameter struct as one vector
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end   % a + al*b
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end   % c*a
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end   % inner product
