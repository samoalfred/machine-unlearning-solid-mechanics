%% Matched-iteration retrained reference for the PINN case studies (Tables 5-6).
% Re-runs Algorithm 1 for Case Studies 5 and 6 exactly as algorithm2_pinn.m
% (same settings, same per-seed RNG, 5 seeds) and adds two references that receive
% the same total number of training iterations as the unlearned model (180 + 150):
%   retrFT   : the standard retrained (clean) model, then the same 150-iteration
%              physics-regularized fine-tune on its own clean data
%   retrCold : a cold retrain on the clean data for 180 + 150 iterations
% It also records whether the fixed-damping Newton step raised the retained
% boundary-data loss (the condition under which Algorithm 1's adaptive-damping
% safeguard would have increased gamma).
% Saves results/matched_retrain_pinn.mat (resumable, checkpoint after every seed).
clear; clc;
L=2; R=0.5; Nc=300; Nb=200; Nbad=40; beta=3e-5;
S=5; iters=180; ft=150; cgIters=50; gammaFrac=1e-3;
if ~exist('results','dir'); mkdir('results'); end
pf=fullfile('results','matched_retrain_pinn.mat');
if exist(pf,'file'), MP=load(pf).MP; else, MP=struct('mse',{cell(1,2)},'trig',{cell(1,2)},'loss',{cell(1,2)}); end
% mse columns: [retrained biased unlearned retrFT retrCold]; loss: [retained data loss at biased, after Newton]
for pidx=1:2
    isH=(pidx==2);
    if isH, layers=[2 30 30 30 2]; mu=1; lam=1; la=0.05; off=[0.15 0.10];
        uref=@(x,y) deal(la*x.*(1+0.5*R^2./(x.^2+y.^2)), -la*y.*(1+0.5*R^2./(x.^2+y.^2)));
    else, layers=[2 50 50 50 2]; E=1; nu=0.33; la=0.1; off=[0.30 0.20];
        uref=@(x,y) deal((la/E)*x.*(1+0.5*R^2./(x.^2+y.^2)), -(nu*la/E)*y.*(1+0.5*R^2./(x.^2+y.^2))); mu=E; lam=nu;
    end
    nL=numel(layers)-1;
    for s=size(MP.mse{pidx},1)+1:S
        rng(60+s+10*pidx);
        Xc=sampleDomain(Nc,L,R); th=2*pi*rand(1,Nb/2); Xh=[R*cos(th);R*sin(th)];
        Xo=outerEdges(Nb-Nb/2,L); Xb=[Xh,Xo]; [a,b]=uref(Xb(1,:),Xb(2,:)); Ub=[a;b];
        bad=randperm(size(Xh,2),Nbad); Ubd=Ub; Ubd(1,bad)=Ubd(1,bad)+off(1); Ubd(2,bad)=Ubd(2,bad)+off(2);
        Xt=sampleDomain(1500,L,R); [tx,ty]=uref(Xt(1,:),Xt(2,:)); Ut=[tx;ty]; keep=true(1,size(Xb,2)); keep(bad)=false;
        pC=trainP(layers,Xc,Xb,Ub,L,iters,beta,mu,lam,isH); pB=trainP(layers,Xc,Xb,Ubd,L,iters,beta,mu,lam,isH);
        gbad=dlfeval(@gradSub,pB,dlarray(Xb(:,bad)/L),dlarray(Ubd(:,bad)),Nb,nL);
        lmax=powerLambda(pB,dlarray(Xb/L),dlarray(Ubd),nL,20); gamma=gammaFrac*lmax;
        v=cg(pB,gbad,dlarray(Xb/L),dlarray(Ubd),gamma,cgIters,1e-6,nL);
        pN=axpy(pB,1.0,v); pA=fineTune(pN,layers,Xc,Xb(:,keep),Ub(:,keep),L,ft,beta,mu,lam,isH);
        lB=retLoss(pB,Xb(:,keep),Ub(:,keep),L,nL); lN=retLoss(pN,Xb(:,keep),Ub(:,keep),L,nL);
        pCF=fineTune(pC,layers,Xc,Xb,Ub,L,ft,beta,mu,lam,isH);          % matched: retrained + same fine-tune
        pCC=trainP(layers,Xc,Xb,Ub,L,iters+ft,beta,mu,lam,isH);         % matched: cold retrain, 330 iterations
        MP.mse{pidx}(s,:)=[pmse(pC,Xt,Ut,nL,L) pmse(pB,Xt,Ut,nL,L) pmse(pA,Xt,Ut,nL,L) pmse(pCF,Xt,Ut,nL,L) pmse(pCC,Xt,Ut,nL,L)];
        MP.loss{pidx}(s,:)=[lB lN]; MP.trig{pidx}(s,1)=lN>lB;
        save(pf,'MP');
        fprintf('CS%d seed %d: retr %.3e biased %.3e unl %.3e retrFT %.3e retrCold %.3e | safeguard trigger %d\n', ...
            pidx+4,s,MP.mse{pidx}(s,:),MP.trig{pidx}(s));
    end
end
fprintf('\n=== means over %d seeds: [retrained biased unlearned retrFT retrCold] ===\n',S);
for pidx=1:2
    M=MP.mse{pidx}; [~,p0]=ttest(M(:,3),M(:,1)); [~,p1]=ttest(M(:,3),M(:,4)); [~,p2]=ttest(M(:,3),M(:,5));
    fprintf('CS%d %s | unl vs retr p=%.3f, vs retrFT p=%.3f, vs retrCold p=%.3f | triggers %d/%d\n', ...
        pidx+4,mat2str(mean(M,1),3),p0,p1,p2,sum(MP.trig{pidx}),S);
end

function l=retLoss(p,Xb,Ub,L,nL); l=double(gather(extractdata(mean((mlp(dlarray(Xb/L),p,nL)-dlarray(Ub)).^2,'all')))); end

%% ---- helpers ----
% mlp: forward pass -- tanh hidden layers, linear output
function Y=mlp(X,p,L); A=X; for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L)); end
% initP: He-style init; for PINNs the last layer starts near zero (zl) for a gentle start
function p=initP(layers,zl); p=struct; nL=numel(layers)-1;
    for i=1:nL, sc=sqrt(2/layers(i)); if zl&&i==nL, sc=0.01; end
        p.(sprintf('W%d',i))=dlarray(sc*randn(layers(i+1),layers(i))); p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1)); end
end
% trainP: PINN training with Adam, minimizing data loss + beta*physics loss
function p=trainP(layers,Xc,Xb,Ub,L,iters,beta,mu,lam,H)
    p=initP(layers,H); Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub); aG=[];aS=[];lr=2e-3;nL=numel(layers)-1;   % inputs scaled by L
    for it=1:iters, [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,L,nL,beta,mu,lam,H); if it==round(0.7*iters),lr=5e-4;end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
% fineTune: like trainP but starts from the given params (used after the influence step)
function p=fineTune(p,layers,Xc,Xb,Ub,L,ft,beta,mu,lam,H)
    Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub); aG=[];aS=[];lr=5e-4;nL=numel(layers)-1;
    for it=1:ft, [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,L,nL,beta,mu,lam,H); [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
% loss: boundary data MSE + beta * mean squared equilibrium residual
function [l,grad]=loss(p,Xc,Xb,Ub,L,nL,beta,mu,lam,H)
    dl=mean((mlp(Xb,p,nL)-Ub).^2,'all');   % boundary data term
    if H, [rx,ry]=eqH(Xc,p,L,nL,mu,lam); else, [rx,ry]=eqL(Xc,p,L,nL,mu,lam); end   % equilibrium residual (Neo-Hookean or linear)
    l=dl+beta*mean(rx.^2+ry.^2,'all'); grad=dlgradient(l,p);
end
% gradSub: g_bad = gradient of (sum of subset squared errors)/Nb (boundary data loss)
function g=gradSub(p,Xs,Ts,Nb,nL); l=sum((mlp(Xs,p,nL)-Ts).^2,'all')/Nb; g=dlgradient(l,p); end
% hvp: Hessian-vector product of the boundary DATA loss, via double backprop (no H formed)
function Hv=hvp(p,v,Xb,Ub,nL)
    g=dlgradient(mean((mlp(Xb,p,nL)-Ub).^2,'all'),p,'EnableHigherDerivatives',true);
    gv=0; f=fieldnames(g); for i=1:numel(f), gv=gv+sum(g.(f{i}).*v.(f{i}),'all'); end; Hv=dlgradient(gv,p);
end
% powerLambda: power iteration (through hvp) for the top data-Hessian eigenvalue
function lam=powerLambda(p,Xb,Ub,nL,it0); v=randLike(p); v=nrm(v);
    for it=1:it0, Hv=dlfeval(@hvp,p,v,Xb,Ub,nL); lam=sdot(v,Hv); v=nrm(Hv); end
end
% cg: damped conjugate gradient for (H_data + lam I) v = g, with a curvature guard
function v=cg(p,g,Xb,Ub,lam,maxit,tol,nL)
    x=scal(g,0); r=g; pp=r; rs=sdot(r,r); g0=sqrt(rs);
    for it=1:maxit, Hp=dlfeval(@hvp,p,pp,Xb,Ub,nL); Ap=axpy(Hp,lam,pp);   % Ap = (H + lam I) pp
        den=sdot(pp,Ap); if den<=0, den=den+lam*sdot(pp,pp)+1e-12; end   % guard against non-positive curvature
        al=rs/den; x=axpy(x,al,pp); r=axpy(r,-al,Ap); rs2=sdot(r,r); if sqrt(rs2)<tol*g0, break; end
        pp=axpy(r,rs2/rs,pp); rs=rs2; end
    v=x;
end
% eqL: LINEAR elasticity equilibrium residual div(sigma), via automatic differentiation
function [rx,ry]=eqL(Xs,p,L,nL,E,nu)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L; duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;   % displacement gradients
    exx=dux(1,:); eyy=duy(2,:); gxy=dux(2,:)+duy(1,:); c=E/(1-nu^2); G=E/(2*(1+nu));   % strains + plane-stress constants
    sxx=c*(exx+nu*eyy); syy=c*(eyy+nu*exx); sxy=G*gxy;   % stresses (plane-stress constitutive law)
    dsxx=dlgradient(sum(sxx,'all'),Xs,'EnableHigherDerivatives',true)/L; dsyy=dlgradient(sum(syy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dsxy=dlgradient(sum(sxy,'all'),Xs,'EnableHigherDerivatives',true)/L; rx=dsxx(1,:)+dsxy(2,:); ry=dsxy(1,:)+dsyy(2,:);   % residual = divergence of stress
end
% eqH: NEO-HOOKEAN equilibrium residual div(P), P = first Piola-Kirchhoff stress, via AD
function [rx,ry]=eqH(Xs,p,L,nL,mu,lam)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L; duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    a=1+dux(1,:); b=dux(2,:); c=duy(1,:); d=1+duy(2,:); J=a.*d-b.*c;   % deformation gradient F and its determinant J
    iTxx=d./J; iTxy=-c./J; iTyx=-b./J; iTyy=a./J; lJ=log(J);   % inverse-transpose of F and log J
    Pxx=mu*(a-iTxx)+lam*lJ.*iTxx; Pxy=mu*(b-iTxy)+lam*lJ.*iTxy; Pyx=mu*(c-iTyx)+lam*lJ.*iTyx; Pyy=mu*(d-iTyy)+lam*lJ.*iTyy;   % P = mu(F - F^-T) + lam*lnJ*F^-T
    dPxx=dlgradient(sum(Pxx,'all'),Xs,'EnableHigherDerivatives',true)/L; dPxy=dlgradient(sum(Pxy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dPyx=dlgradient(sum(Pyx,'all'),Xs,'EnableHigherDerivatives',true)/L; dPyy=dlgradient(sum(Pyy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    rx=dPxx(1,:)+dPxy(2,:); ry=dPyx(1,:)+dPyy(2,:);   % residual = divergence of P
end
% pmse: test MSE against the analytic reference field (inputs scaled by L)
function m=pmse(p,Xt,Ut,nL,L); U=extractdata(mlp(dlarray(Xt/L),p,nL)); m=mean((U(:)-Ut(:)).^2); end
% meanRes/resField: mean |residual| over points, and the per-point residual magnitude
function rm=meanRes(p,Xc,mu,lam,L,nL,H); rm=mean(resField(p,Xc,mu,lam,L,nL,H)); end
function r=resField(p,Xg,mu,lam,L,nL,H)
    Xs=dlarray(Xg/L); if H, [rx,ry]=dlfeval(@eqH,Xs,p,L,nL,mu,lam); else, [rx,ry]=dlfeval(@eqL,Xs,p,L,nL,mu,lam); end
    r=sqrt(extractdata(rx).^2+extractdata(ry).^2);
end
% sampleDomain: rejection-sample N interior points that lie outside the hole
function X=sampleDomain(N,L,R); X=zeros(2,N); c=0; while c<N, q=-L+2*L*rand(2,1); if norm(q)>R,c=c+1;X(:,c)=q;end, end, end
% outerEdges: N points sampled uniformly along the four outer edges of the plate
function X=outerEdges(N,L); X=zeros(2,N);
    for i=1:N, s=randi(4); t=-L+2*L*rand(); switch s, case 1,X(:,i)=[t;L]; case 2,X(:,i)=[t;-L]; case 3,X(:,i)=[L;t]; case 4,X(:,i)=[-L;t]; end, end
end
% struct-vector algebra: treat the parameter struct as one vector (for CG/power iteration)
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end   % a + al*b
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end   % c*a
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end   % inner product
function v=randLike(p); v=p; f=fieldnames(p); for i=1:numel(f), v.(f{i})=dlarray(randn(size(p.(f{i})))); end, end   % random vector shaped like p
function v=nrm(v); n=sqrt(sdot(v,v)); f=fieldnames(v); for i=1:numel(f), v.(f{i})=v.(f{i})/n; end, end   % normalize to unit length
