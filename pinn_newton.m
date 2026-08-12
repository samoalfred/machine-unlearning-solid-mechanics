%% Second-order (Newton) influence unlearning on the Neo-Hookean PINN.
% Hardest case: finite-deformation hyperelasticity, nondimensional. The
% corrupted data are boundary displacements, so the removal direction is a
% Newton step on the boundary DATA loss:
%     theta_un = theta_hat + (H_data + gamma I)^{-1} g_bad,
% where H_data is the Hessian of the total boundary data loss (a PSD
% Gauss-Newton-friendly object, no input derivatives) and g_bad is the
% corrupted subset's contribution. Solved by conjugate gradient with
% automatic-differentiation Hessian-vector products. Evaluated WITH NO
% fine-tuning. The equilibrium residual |div P| is still computed by AD as a
% diagnostic. Compared to biased, retrained (clean), and the first-order step.

clear; clc; close all;
rng(42);
L=2; R=0.5; mu=1.0; lambda=1.0; load_amp=0.05;
N_coll=300; N_bnd=200; N_bad=40; layers=[2 30 30 30 2];
iters=250; beta=3e-5; nSeed=2;
lams=[1e-2 1e-1 1];
uref=@(x,y) deal(load_amp*x.*(1+0.5*R^2./(x.^2+y.^2)), -load_amp*y.*(1+0.5*R^2./(x.^2+y.^2)));

mse_b=zeros(nSeed,1); mse_r=zeros(nSeed,1); mse_f=zeros(nSeed,1);
res_b=zeros(nSeed,1); res_r=zeros(nSeed,1);
mse_n=zeros(nSeed,numel(lams)); res_n=zeros(nSeed,numel(lams)); sn=zeros(nSeed,numel(lams));

for s=1:nSeed
    rng(60+s);
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

    mse_b(s)=predMSE(pB,Xt,Ut,layers,L); res_b(s)=meanResidual(pB,Xc,mu,lambda,L,layers);
    mse_r(s)=predMSE(pC,Xt,Ut,layers,L); res_r(s)=meanResidual(pC,Xc,mu,lambda,L,layers);

    % RHS: corrupted subset contribution to total boundary mean data loss
    Xbd=dlarray(Xb/L); Ubd=dlarray(Ub_bad);
    gbad=dlfeval(@gradSubset, pB, dlarray(Xb(:,badIdx)/L), dlarray(Ub_bad(:,badIdx)), N_bnd, nL);

    % first-order step (contrast)
    pF=axpy(pB,-1e-6,gbad);
    mse_f(s)=predMSE(pF,Xt,Ut,layers,L);

    for li=1:numel(lams)
        v=cgSolve(pB,gbad,Xbd,Ubd,lams(li),100,1e-6,nL);
        pN=axpy(pB,1.0,v);
        mse_n(s,li)=predMSE(pN,Xt,Ut,layers,L);
        res_n(s,li)=meanResidual(pN,Xc,mu,lambda,L,layers);
        sn(s,li)=sqrt(sdot(v,v));
    end
    fprintf('seed %d: biased %.3e retrain %.3e first %.3e | newton ',s,mse_b(s),mse_r(s),mse_f(s));
    fprintf('%.3e ',mse_n(s,:)); fprintf('\n');
end

fprintf('\n=== Newton influence on Neo-Hookean PINN (mean over %d seeds), NO fine-tuning ===\n',nSeed);
fprintf('Biased    : MSE %.3e  resid %.3f\n',mean(mse_b),mean(res_b));
fprintf('Retrained : MSE %.3e  resid %.3f\n',mean(mse_r),mean(res_r));
fprintf('First-ord : MSE %.3e\n',mean(mse_f));
for li=1:numel(lams)
    fprintf('Newton lam=%.0e: MSE %.3e  resid %.3f  ||step|| %.3e\n', ...
        lams(li),mean(mse_n(:,li)),mean(res_n(:,li)),mean(sn(:,li)));
end
if ~exist('results','dir'); mkdir('results'); end
hp=struct('biased_mse',mean(mse_b),'biased_res',mean(res_b),'retr_mse',mean(mse_r),'retr_res',mean(res_r), ...
    'first_mse',mean(mse_f),'lams',lams,'newton_mse',mean(mse_n,1),'newton_res',mean(res_n,1),'stepnorm',mean(sn,1));
save(fullfile('results','pinn_newton.mat'),'hp');
fprintf('\nSaved results/pinn_newton.mat\n');

%% helpers
function X=sampleDomain(N,L,R); X=zeros(2,N); c=0;
    while c<N, p=-L+2*L*rand(2,1); if norm(p)>R, c=c+1; X(:,c)=p; end, end
end
function X=outerEdges(N,L); X=zeros(2,N);
    for i=1:N, s=randi(4); t=-L+2*L*rand();
        switch s, case 1,X(:,i)=[t;L]; case 2,X(:,i)=[t;-L]; case 3,X(:,i)=[L;t]; case 4,X(:,i)=[-L;t]; end
    end
end
function Y=mlp(X,p,L); A=X;
    for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L));
end
function p=initP(layers); p=struct;
    for i=1:numel(layers)-1
        s=sqrt(2/layers(i)); if i==numel(layers)-1, s=0.01; end
        p.(sprintf('W%d',i))=dlarray(s*randn(layers(i+1),layers(i)));
        p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1));
    end
end
function p=trainPINN(layers,Xc,Xb,Ub,mu,lam,L,iters,beta)
    p=initP(layers); Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub);
    aG=[]; aS=[]; lr=2e-3; nL=numel(layers)-1;
    for it=1:iters
        [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,mu,lam,L,nL,beta);
        if it==round(0.7*iters), lr=5e-4; end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr);
    end
end
function [l,grad]=loss(p,Xc,Xb,Ub,mu,lam,L,nL,beta)
    dl=mean((mlp(Xb,p,nL)-Ub).^2,'all');
    [rx,ry]=equilibrium(Xc,p,mu,lam,L,nL);
    l=dl+beta*mean(rx.^2+ry.^2,'all'); grad=dlgradient(l,p);
end
function g=gradSubset(p,Xsub,Tsub,Nbnd,nL)
    l=sum((mlp(Xsub,p,nL)-Tsub).^2,'all')/Nbnd; g=dlgradient(l,p);
end
function Hv=hvpData(p,v,Xb,Ub,nL)
    g=dlgradient(mean((mlp(Xb,p,nL)-Ub).^2,'all'), p, 'EnableHigherDerivatives',true);
    gv=0; f=fieldnames(g); for i=1:numel(f), gv=gv+sum(g.(f{i}).*v.(f{i}),'all'); end
    Hv=dlgradient(gv,p);
end
function v=cgSolve(p,g,Xb,Ub,lam,maxit,tol,nL)
    x=scal(g,0); r=g; pp=r; rs=sdot(r,r); g0=sqrt(rs);
    for it=1:maxit
        Hp=dlfeval(@hvpData,p,pp,Xb,Ub,nL);
        Ap=axpy(Hp,lam,pp);
        den=sdot(pp,Ap); if den<=0, den=den+lam*sdot(pp,pp)+1e-12; end
        al=rs/den; x=axpy(x,al,pp); r=axpy(r,-al,Ap);
        rs2=sdot(r,r); if sqrt(rs2)<tol*g0, break; end
        pp=axpy(r,rs2/rs,pp); rs=rs2;
    end
    v=x;
end
function [rx,ry]=equilibrium(Xs,p,mu,lam,L,nL)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L;
    duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    a=1+dux(1,:); b=dux(2,:); c=duy(1,:); d=1+duy(2,:);
    J=a.*d-b.*c; iTxx=d./J; iTxy=-c./J; iTyx=-b./J; iTyy=a./J; lJ=log(J);
    Pxx=mu*(a-iTxx)+lam*lJ.*iTxx; Pxy=mu*(b-iTxy)+lam*lJ.*iTxy;
    Pyx=mu*(c-iTyx)+lam*lJ.*iTyx; Pyy=mu*(d-iTyy)+lam*lJ.*iTyy;
    dPxx=dlgradient(sum(Pxx,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dPxy=dlgradient(sum(Pxy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dPyx=dlgradient(sum(Pyx,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dPyy=dlgradient(sum(Pyy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    rx=dPxx(1,:)+dPxy(2,:); ry=dPyx(1,:)+dPyy(2,:);
end
function m=predMSE(p,Xt,Ut,layers,L); U=extractdata(mlp(dlarray(Xt/L),p,numel(layers)-1)); m=mean((U(:)-Ut(:)).^2); end
function rm=meanResidual(p,Xc,mu,lam,L,layers)
    Xs=dlarray(Xc/L); [rx,ry]=dlfeval(@equilibrium,Xs,p,mu,lam,L,numel(layers)-1);
    rm=mean(sqrt(extractdata(rx).^2+extractdata(ry).^2));
end
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end
