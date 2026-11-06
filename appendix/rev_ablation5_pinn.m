%% Revision: 5-seed version of ablation_all_pinn.m with a matched baseline.
%% Extra ACC column 7 = clean model followed by the same fine-tune (matched iteration count).
%% Run: for ABP_CASE=1:2, rev_ablation5_pinn; end   -> results/ablation5_pinn.mat
%% Ablation on the PINN case studies (CS5-6), two knobs:
%%   (A) number of corrupted/removed hole-boundary nodes; (B) corruption magnitude.
%% Reduced training budget (trends, not headline numbers). 2 seeds. Incremental:
%% set ABP_CASE (1->CS5, 2->CS6); appends to results/ablation_all_pinn.mat.
if ~exist('results','dir'); mkdir('results'); end
Ld=2; R=0.5; Nc=150; Nb=200; beta=3e-5; iters=130; ft=90; cgIters=40; gammaFrac=1e-3; S=5;
nList=[10 20 40 60]; kList=[1 2 3 5];
accf=fullfile('results','ablation5_pinn.mat');
if exist(accf,'file'), ACC=load(accf).ACC; else, ACC=[]; end
% ACC cols: [case(5/6) knob(1=samples,2=mag) value seed postMSE cleanMSE]

if ~exist('ABP_CASE','var') || isempty(ABP_CASE); printAbP(ACC); return; end
pidx=ABP_CASE; isH=(pidx==2); cs=pidx+4;
if isH, layers=[2 30 30 30 2]; mu=1; lam=1; off0=[0.15 0.10];
    uref=@(x,y) deal(0.05*x.*(1+0.5*R^2./(x.^2+y.^2)), -0.05*y.*(1+0.5*R^2./(x.^2+y.^2)));
else, layers=[2 50 50 50 2]; E=1; nu=0.33; off0=[0.30 0.20]; mu=E; lam=nu;
    uref=@(x,y) deal((0.1/E)*x.*(1+0.5*R^2./(x.^2+y.^2)), -(nu*0.1/E)*y.*(1+0.5*R^2./(x.^2+y.^2)));
end
nL=numel(layers)-1;
have=@(kn,v,s) ~isempty(ACC) && any(ACC(:,1)==cs&ACC(:,2)==kn&abs(ACC(:,3)-v)<1e-9&ACC(:,4)==s);

for s=1:S
    rng(8000+10*pidx+s);
    Xc=sampleDomain(Nc,Ld,R); th=2*pi*rand(1,Nb/2); Xh=[R*cos(th);R*sin(th)];
    Xo=outerEdges(Nb-Nb/2,Ld); Xb=[Xh,Xo]; [a,b]=uref(Xb(1,:),Xb(2,:)); Ub=[a;b];
    Xt=sampleDomain(1000,Ld,R); [tx,ty]=uref(Xt(1,:),Xt(2,:)); Ut=[tx;ty];
    % clean baseline
    pCb=trainP(layers,Xc,Xb,Ub,Ld,iters,beta,mu,lam,isH); cleanMSE=pmse(pCb,Xt,Ut,nL,Ld); cleanFT=pmse(fineTune(pCb,layers,Xc,Xb,Ub,Ld,ft,beta,mu,lam,isH),Xt,Ut,nL,Ld);
    % (A) samples sweep (nominal magnitude)
    for nb=nList
        if have(1,nb,s), continue; end
        m=runPinn(layers,nL,Xc,Xb,Ub,Xh,Xt,Ut,nb,1,off0,Ld,iters,ft,beta,mu,lam,isH,gammaFrac,cgIters);
        ACC=[ACC; cs 1 nb s m cleanMSE cleanFT]; save(accf,'ACC');
        fprintf('CS%d A nb=%d s=%d: post %.3e (clean %.3e)\n',cs,nb,s,m,cleanMSE);
    end
    % (B) magnitude sweep (40 nodes)
    for k=kList
        if have(2,k,s), continue; end
        m=runPinn(layers,nL,Xc,Xb,Ub,Xh,Xt,Ut,40,k,off0,Ld,iters,ft,beta,mu,lam,isH,gammaFrac,cgIters);
        ACC=[ACC; cs 2 k s m cleanMSE cleanFT]; save(accf,'ACC');
        fprintf('CS%d B k=%g s=%d: post %.3e\n',cs,k,s,m);
    end
end
fprintf('CS%d ablation done.\n',cs);

%% ---- one PINN pipeline, returns field MSE ----
function m=runPinn(layers,nL,Xc,Xb,Ub,Xh,Xt,Ut,nbad,kap,off0,L,iters,ft,beta,mu,lam,isH,gammaFrac,cgIters)
    off=kap*off0; bad=randperm(size(Xh,2),min(nbad,size(Xh,2)));
    Ubd=Ub; Ubd(1,bad)=Ubd(1,bad)+off(1); Ubd(2,bad)=Ubd(2,bad)+off(2);
    keep=true(1,size(Xb,2)); keep(bad)=false;
    pB=trainP(layers,Xc,Xb,Ubd,L,iters,beta,mu,lam,isH);
    gbad=dlfeval(@gradSub,pB,dlarray(Xb(:,bad)/L),dlarray(Ubd(:,bad)),size(Xb,2),nL);
    pN=adaptiveNewtonP(pB,dlarray(Xb/L),dlarray(Ubd),dlarray(Xb(:,keep)/L),dlarray(Ub(:,keep)),gbad,gammaFrac,cgIters,nL);
    pA=fineTune(pN,layers,Xc,Xb(:,keep),Ub(:,keep),L,ft,beta,mu,lam,isH);
    m=pmse(pA,Xt,Ut,nL,L);
end
function pN=adaptiveNewtonP(pB,Xb,Ub,Xkeep,Ukeep,g,gf0,cgIters,nL)
    lam=powerLambda(pB,Xb,Ub,nL,20);
    Lb=double(gather(extractdata(mean((mlp(Xkeep,pB,nL)-Ukeep).^2,'all')))); gf=gf0; cand=pB;
    for tryi=1:7
        v=cg(pB,g,Xb,Ub,gf*lam,cgIters,1e-6,nL); cand=axpy(pB,1.0,v);
        Lc=double(gather(extractdata(mean((mlp(Xkeep,cand,nL)-Ukeep).^2,'all'))));
        if isfinite(Lc)&&Lc<5*Lb, pN=cand; return; end
        gf=gf*10;
    end
    pN=cand;
end
function printAbP(ACC)
    if isempty(ACC), fprintf('No PINN ablation yet. Set ABP_CASE=1 and run.\n'); return; end
    nm=containers.Map({5,6},{'Elasticity PINN','Neo-Hookean PINN'});
    fprintf('\n=== PINN ablation (post-unlearning field MSE, mean over seeds, reduced scale) ===\n');
    for cs=[5 6]
        A=ACC(ACC(:,1)==cs & ACC(:,2)==1,:); if isempty(A), continue; end
        fprintf('CS%d %-16s | samples:',cs,nm(cs));
        for nb=unique(A(:,3))', fprintf(' n=%d %.2e',nb,mean(A(A(:,3)==nb,5))); end; fprintf('\n');
        B=ACC(ACC(:,1)==cs & ACC(:,2)==2,:);
        fprintf('                       | magnitude:');
        for k=unique(B(:,3))', fprintf(' x%g %.2e',k,mean(B(B(:,3)==k,5))); end; fprintf('\n');
    end
end
%% ---- PINN helpers (from rev_pinn_fourway.m) ----
function Y=mlp(X,p,L); A=X; for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L)); end
function p=initP(layers,zl); p=struct; nL=numel(layers)-1;
    for i=1:nL, sc=sqrt(2/layers(i)); if zl&&i==nL, sc=0.01; end
        p.(sprintf('W%d',i))=dlarray(sc*randn(layers(i+1),layers(i))); p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1)); end
end
function p=trainP(layers,Xc,Xb,Ub,L,iters,beta,mu,lam,H)
    p=initP(layers,H); Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub); aG=[];aS=[];lr=2e-3;nL=numel(layers)-1;
    for it=1:iters, [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,L,nL,beta,mu,lam,H); if it==round(0.7*iters),lr=5e-4;end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
function p=fineTune(p,layers,Xc,Xb,Ub,L,ft,beta,mu,lam,H)
    Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub); aG=[];aS=[];lr=5e-4;nL=numel(layers)-1;
    for it=1:ft, [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,L,nL,beta,mu,lam,H); [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
function [l,grad]=loss(p,Xc,Xb,Ub,L,nL,beta,mu,lam,H)
    dl=mean((mlp(Xb,p,nL)-Ub).^2,'all');
    if H, [rx,ry]=eqH(Xc,p,L,nL,mu,lam); else, [rx,ry]=eqL(Xc,p,L,nL,mu,lam); end
    l=dl+beta*mean(rx.^2+ry.^2,'all'); grad=dlgradient(l,p);
end
function g=gradSub(p,Xs,Ts,Nb,nL); l=sum((mlp(Xs,p,nL)-Ts).^2,'all')/Nb; g=dlgradient(l,p); end
function Hv=hvp(p,v,Xb,Ub,nL)
    g=dlgradient(mean((mlp(Xb,p,nL)-Ub).^2,'all'),p,'EnableHigherDerivatives',true);
    gv=0; f=fieldnames(g); for i=1:numel(f), gv=gv+sum(g.(f{i}).*v.(f{i}),'all'); end; Hv=dlgradient(gv,p);
end
function lam=powerLambda(p,Xb,Ub,nL,it0); v=randLike(p); v=nrm(v);
    for it=1:it0, Hv=dlfeval(@hvp,p,v,Xb,Ub,nL); lam=sdot(v,Hv); v=nrm(Hv); end
end
function v=cg(p,g,Xb,Ub,lam,maxit,tol,nL)
    x=scal(g,0); r=g; pp=r; rs=sdot(r,r); g0=sqrt(rs);
    for it=1:maxit, Hp=dlfeval(@hvp,p,pp,Xb,Ub,nL); Ap=axpy(Hp,lam,pp);
        den=sdot(pp,Ap); if den<=0, den=den+lam*sdot(pp,pp)+1e-12; end
        al=rs/den; x=axpy(x,al,pp); r=axpy(r,-al,Ap); rs2=sdot(r,r); if sqrt(rs2)<tol*g0, break; end
        pp=axpy(r,rs2/rs,pp); rs=rs2; end
    v=x;
end
function [rx,ry]=eqL(Xs,p,L,nL,E,nu)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L; duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    exx=dux(1,:); eyy=duy(2,:); gxy=dux(2,:)+duy(1,:); c=E/(1-nu^2); G=E/(2*(1+nu));
    sxx=c*(exx+nu*eyy); syy=c*(eyy+nu*exx); sxy=G*gxy;
    dsxx=dlgradient(sum(sxx,'all'),Xs,'EnableHigherDerivatives',true)/L; dsyy=dlgradient(sum(syy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dsxy=dlgradient(sum(sxy,'all'),Xs,'EnableHigherDerivatives',true)/L; rx=dsxx(1,:)+dsxy(2,:); ry=dsxy(1,:)+dsyy(2,:);
end
function [rx,ry]=eqH(Xs,p,L,nL,mu,lam)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L; duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    a=1+dux(1,:); b=dux(2,:); c=duy(1,:); d=1+duy(2,:); J=a.*d-b.*c;
    iTxx=d./J; iTxy=-c./J; iTyx=-b./J; iTyy=a./J; lJ=log(J);
    Pxx=mu*(a-iTxx)+lam*lJ.*iTxx; Pxy=mu*(b-iTxy)+lam*lJ.*iTxy; Pyx=mu*(c-iTyx)+lam*lJ.*iTyx; Pyy=mu*(d-iTyy)+lam*lJ.*iTyy;
    dPxx=dlgradient(sum(Pxx,'all'),Xs,'EnableHigherDerivatives',true)/L; dPxy=dlgradient(sum(Pxy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dPyx=dlgradient(sum(Pyx,'all'),Xs,'EnableHigherDerivatives',true)/L; dPyy=dlgradient(sum(Pyy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    rx=dPxx(1,:)+dPxy(2,:); ry=dPyx(1,:)+dPyy(2,:);
end
function m=pmse(p,Xt,Ut,nL,L); U=extractdata(mlp(dlarray(Xt/L),p,nL)); m=mean((U(:)-Ut(:)).^2); end
function X=sampleDomain(N,L,R); X=zeros(2,N); c=0; while c<N, q=-L+2*L*rand(2,1); if norm(q)>R,c=c+1;X(:,c)=q;end, end, end
function X=outerEdges(N,L); X=zeros(2,N);
    for i=1:N, s=randi(4); t=-L+2*L*rand(); switch s, case 1,X(:,i)=[t;L]; case 2,X(:,i)=[t;-L]; case 3,X(:,i)=[L;t]; case 4,X(:,i)=[-L;t]; end, end
end
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end
function v=randLike(p); v=p; f=fieldnames(p); for i=1:numel(f), v.(f{i})=dlarray(randn(size(p.(f{i})))); end, end
function v=nrm(v); n=sqrt(sdot(v,v)); f=fieldnames(v); for i=1:numel(f), v.(f{i})=v.(f{i})/n; end, end
