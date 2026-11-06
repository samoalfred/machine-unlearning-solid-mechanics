%% Revision: PINN convergence study (Case Study 5, 2D linear elasticity).
%% (a) Training-iteration sweep at fixed collocation density; (b) collocation
%% sweep at fixed iterations. Reports field MSE (to analytic reference) and mean
%% equilibrium residual, to show the PINN converges and that the unlearning
%% conclusion is not an artifact of under-training. (c) At the largest budget,
%% the full biased->Newton+fine-tune pipeline is compared to retraining.
clear; clc; close all;
Ld=2; R=0.5; Nb=200; Nbad=40; beta=3e-5; off=[0.30 0.20];
layers=[2 50 50 50 2]; mu=1; lam=0.33; E=1; nu=0.33; la=0.1; nL=numel(layers)-1; isH=false;
uref=@(x,y) deal((la/E)*x.*(1+0.5*R^2./(x.^2+y.^2)), -(nu*la/E)*y.*(1+0.5*R^2./(x.^2+y.^2)));
Sseed=2;
iterList=[150 300 600 1200]; colloList=[150 300 600 1200];
MSE_it=zeros(Sseed,numel(iterList)); RES_it=zeros(Sseed,numel(iterList));
MSE_co=zeros(Sseed,numel(colloList)); RES_co=zeros(Sseed,numel(colloList));
for s=1:Sseed
  % (a) iteration sweep, fixed Nc=300
  for i=1:numel(iterList)
    rng(500+s); Nc=300;
    Xc=sampleDomain(Nc,Ld,R); th=2*pi*rand(1,Nb/2); Xh=[R*cos(th);R*sin(th)];
    Xo=outerEdges(Nb-Nb/2,Ld); Xb=[Xh,Xo]; [a,b]=uref(Xb(1,:),Xb(2,:)); Ub=[a;b];
    Xt=sampleDomain(1500,Ld,R); [tx,ty]=uref(Xt(1,:),Xt(2,:)); Ut=[tx;ty];
    pC=trainP(layers,Xc,Xb,Ub,Ld,iterList(i),beta,mu,lam,isH);
    MSE_it(s,i)=pmse(pC,Xt,Ut,nL,Ld); RES_it(s,i)=meanRes(pC,Xc,mu,lam,Ld,nL,isH);
  end
  % (b) collocation sweep, fixed iters=600
  for i=1:numel(colloList)
    rng(500+s); Nc=colloList(i);
    Xc=sampleDomain(Nc,Ld,R); th=2*pi*rand(1,Nb/2); Xh=[R*cos(th);R*sin(th)];
    Xo=outerEdges(Nb-Nb/2,Ld); Xb=[Xh,Xo]; [a,b]=uref(Xb(1,:),Xb(2,:)); Ub=[a;b];
    Xt=sampleDomain(1500,Ld,R); [tx,ty]=uref(Xt(1,:),Xt(2,:)); Ut=[tx;ty];
    pC=trainP(layers,Xc,Xb,Ub,Ld,600,beta,mu,lam,isH);
    MSE_co(s,i)=pmse(pC,Xt,Ut,nL,Ld); RES_co(s,i)=meanRes(pC,Xc,mu,lam,Ld,nL,isH);
  end
  fprintf('conv seed %d done\n',s);
end
% (c) unlearning at the largest budget (iters=1200, Nc=600)
rng(777); Nc=600; iters=1200; ft=400;
Xc=sampleDomain(Nc,Ld,R); th=2*pi*rand(1,Nb/2); Xh=[R*cos(th);R*sin(th)];
Xo=outerEdges(Nb-Nb/2,Ld); Xb=[Xh,Xo]; [a,b]=uref(Xb(1,:),Xb(2,:)); Ub=[a;b];
bad=randperm(size(Xh,2),Nbad); Ubd=Ub; Ubd(1,bad)=Ubd(1,bad)+off(1); Ubd(2,bad)=Ubd(2,bad)+off(2);
keep=true(1,size(Xb,2)); keep(bad)=false;
Xt=sampleDomain(1500,Ld,R); [tx,ty]=uref(Xt(1,:),Xt(2,:)); Ut=[tx;ty];
pB=trainP(layers,Xc,Xb,Ubd,Ld,iters,beta,mu,lam,isH);
pC=trainP(layers,Xc,Xb,Ub,Ld,iters,beta,mu,lam,isH);
gbad=dlfeval(@gradSub,pB,dlarray(Xb(:,bad)/Ld),dlarray(Ubd(:,bad)),Nb,nL);
lmax=powerLambda(pB,dlarray(Xb/Ld),dlarray(Ubd),nL,20); gamma=1e-3*lmax;
v=cg(pB,gbad,dlarray(Xb/Ld),dlarray(Ubd),gamma,60,1e-6,nL); pN=axpy(pB,1.0,v);
pNF=fineTune(pN,layers,Xc,Xb(:,keep),Ub(:,keep),Ld,ft,beta,mu,lam,isH);
conv_big=struct('biased',pmse(pB,Xt,Ut,nL,Ld),'unlearned',pmse(pNF,Xt,Ut,nL,Ld),'retrain',pmse(pC,Xt,Ut,nL,Ld), ...
  'iters',iters,'Nc',Nc);
if ~exist('results','dir'); mkdir('results'); end
CV=struct('iterList',iterList,'colloList',colloList,'MSE_it',MSE_it,'RES_it',RES_it,'MSE_co',MSE_co,'RES_co',RES_co,'big',conv_big);
save(fullfile('results','rev_pinn_convergence.mat'),'CV');
fprintf('\n=== (a) iteration sweep (Nc=300), field MSE / residual (mean) ===\n');
for i=1:numel(iterList), fprintf('iters %-5d  MSE %.3e  res %.3f\n',iterList(i),mean(MSE_it(:,i)),mean(RES_it(:,i))); end
fprintf('=== (b) collocation sweep (iters=600) ===\n');
for i=1:numel(colloList), fprintf('Nc %-5d    MSE %.3e  res %.3f\n',colloList(i),mean(MSE_co(:,i)),mean(RES_co(:,i))); end
fprintf('=== (c) unlearning at large budget (iters=%d,Nc=%d): biased %.3e unlearned %.3e retrain %.3e ===\n', ...
  conv_big.iters,conv_big.Nc,conv_big.biased,conv_big.unlearned,conv_big.retrain);
fprintf('Saved results/rev_pinn_convergence.mat\n');

%% ---- shared PINN helpers ----
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
function rm=meanRes(p,Xc,mu,lam,L,nL,H); rm=mean(resField(p,Xc,mu,lam,L,nL,H)); end
function r=resField(p,Xg,mu,lam,L,nL,H)
    Xs=dlarray(Xg/L); if H, [rx,ry]=dlfeval(@eqH,Xs,p,L,nL,mu,lam); else, [rx,ry]=dlfeval(@eqL,Xs,p,L,nL,mu,lam); end
    r=sqrt(extractdata(rx).^2+extractdata(ry).^2);
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
