%% Optimizer-effect study on the PINNs (CS5-6): Adam vs Levenberg-Marquardt.
% LM here = damped Newton / Gauss-Newton iteration on the FULL PINN loss
% (boundary data + beta*physics residual) with a Marquardt-style damping loop
% (increase lambda if a step fails to decrease the loss). Because the physics
% residual is itself a 2nd derivative of the network, the Hessian-vector product
% needs 3rd-order nested AD; the problem is run at REDUCED scale to keep it
% tractable, and Adam is trained at the SAME reduced scale for a fair comparison.
%
% Modes (set in base workspace):
%   LM_PROBE=true   -> time a single full-loss HVP and return (feasibility check)
%   ROP_CASE=1|2    -> run that case (1->CS5 LinElastic, 2->CS6 Neo-Hookean)
% Incremental: appends to results/rev_optimizer_pinn.mat.

if ~exist('results','dir'); mkdir('results'); end
Ld=2; R=0.5; Nc=80; Nb=100; Nbad=20; beta=3e-5;
adamIters=150; ft=100; cgIters=12; gammaFrac=1e-3; lmIters=8; S=1;
layersL=[2 20 20 2]; layersH=[2 20 20 2];   % reduced scale

%% ---- probe: one full-loss HVP timing ----
if exist('LM_PROBE','var') && LM_PROBE
    rng(1); layers=layersL; nL=numel(layers)-1; isH=false; mu=1; lam=0.1;
    Xc=sampleDomain(Nc,Ld,R); p=initP(layers,false); v=randLike(p);
    Xcd=dlarray(Xc/Ld); Xbd=dlarray(outerEdges(Nb,Ld)/Ld); Ubd=dlarray(zeros(2,Nb));
    t=tic; Hv=dlfeval(@hvpFull,p,v,Xcd,Xbd,Ubd,Ld,nL,beta,mu,lam,isH); el=toc(t);
    fprintf('full-loss HVP ok: %.3f s, |Hv|=%.3e\n',el,sqrt(sdot(Hv,Hv))); return;
end

if ~exist('ROP_CASE','var') || isempty(ROP_CASE); printPINN(); return; end
pidx=ROP_CASE; isH=(pidx==2);
if isH, layers=layersH; mu=1; lam=1; la=0.05; off=[0.15 0.10]; nm='Neo-Hookean';
    uref=@(x,y) deal(la*x.*(1+0.5*R^2./(x.^2+y.^2)), -la*y.*(1+0.5*R^2./(x.^2+y.^2)));
else, layers=layersL; E=1; nu=0.33; la=0.1; off=[0.30 0.20]; nm='LinElastic';
    uref=@(x,y) deal((la/E)*x.*(1+0.5*R^2./(x.^2+y.^2)), -(nu*la/E)*y.*(1+0.5*R^2./(x.^2+y.^2))); mu=E; lam=nu;
end
nL=numel(layers)-1;
accf=fullfile('results','rev_optimizer_pinn.mat');
if exist(accf,'file'), Ldd=load(accf); ACC=Ldd.ACC; else, ACC=[]; end
% ACC cols: [case seed opt(1=Adam,2=LM) cleanMSE biasedMSE unlearnedMSE]

if ~exist('ROP_OPT','var') || isempty(ROP_OPT); ROP_OPT=[1 2]; end
for s=1:S
    rng(60+s+10*pidx);
    Xc=sampleDomain(Nc,Ld,R); th=2*pi*rand(1,Nb/2); Xh=[R*cos(th);R*sin(th)];
    Xo=outerEdges(Nb-Nb/2,Ld); Xb=[Xh,Xo]; [a,b]=uref(Xb(1,:),Xb(2,:)); Ub=[a;b];
    bad=randperm(size(Xh,2),Nbad); Ubd=Ub; Ubd(1,bad)=Ubd(1,bad)+off(1); Ubd(2,bad)=Ubd(2,bad)+off(2);
    keep=true(1,size(Xb,2)); keep(bad)=false;
    Xt=sampleDomain(1200,Ld,R); [tx,ty]=uref(Xt(1,:),Xt(2,:)); Ut=[tx;ty];

    if any(ROP_OPT==1) && ~(~isempty(ACC)&&any(ACC(:,1)==pidx&ACC(:,2)==s&ACC(:,3)==1))
        pB=trainP(layers,Xc,Xb,Ubd,Ld,adamIters,beta,mu,lam,isH);
        pC=trainP(layers,Xc,Xb,Ub,Ld,adamIters,beta,mu,lam,isH);
        pA=unlearnP(pB,layers,Xc,Xb,Ub,Ubd,keep,bad,Ld,nL,ft,beta,mu,lam,isH,gammaFrac,cgIters);
        ACC=[ACC; pidx s 1 pmse(pC,Xt,Ut,nL,Ld) pmse(pB,Xt,Ut,nL,Ld) pmse(pA,Xt,Ut,nL,Ld)]; save(accf,'ACC');
        fprintf('CS%d seed %d Adam | cln %.2e bia %.2e unl %.2e\n',pidx+4,s,ACC(end,4),ACC(end,5),ACC(end,6));
    end
    if any(ROP_OPT==2) && ~(~isempty(ACC)&&any(ACC(:,1)==pidx&ACC(:,2)==s&ACC(:,3)==2))
        pBl=trainLMpinn(layers,Xc,Xb,Ubd,Ld,lmIters,beta,mu,lam,isH,cgIters,gammaFrac);
        pCl=trainLMpinn(layers,Xc,Xb,Ub,Ld,lmIters,beta,mu,lam,isH,cgIters,gammaFrac);
        pAl=unlearnP(pBl,layers,Xc,Xb,Ub,Ubd,keep,bad,Ld,nL,ft,beta,mu,lam,isH,gammaFrac,cgIters);
        ACC=[ACC; pidx s 2 pmse(pCl,Xt,Ut,nL,Ld) pmse(pBl,Xt,Ut,nL,Ld) pmse(pAl,Xt,Ut,nL,Ld)]; save(accf,'ACC');
        fprintf('CS%d seed %d LM | cln %.2e bia %.2e unl %.2e\n',pidx+4,s,ACC(end,4),ACC(end,5),ACC(end,6));
    end
    if any(ROP_OPT==3) && ~(~isempty(ACC)&&any(ACC(:,1)==pidx&ACC(:,2)==s&ACC(:,3)==3))
        pBg=trainP(layers,Xc,Xb,Ubd,Ld,adamIters,beta,mu,lam,isH);
        pCg=trainP(layers,Xc,Xb,Ub,Ld,adamIters,beta,mu,lam,isH);
        goalB=fullLossVal(pBg,dlarray(Xc/Ld),dlarray(Xb/Ld),dlarray(Ubd),Ld,nL,beta,mu,lam,isH);
        goalC=fullLossVal(pCg,dlarray(Xc/Ld),dlarray(Xb/Ld),dlarray(Ub),Ld,nL,beta,mu,lam,isH);
        pBm=trainLMpinn(layers,Xc,Xb,Ubd,Ld,lmIters,beta,mu,lam,isH,cgIters,gammaFrac,goalB);
        pCm=trainLMpinn(layers,Xc,Xb,Ub,Ld,lmIters,beta,mu,lam,isH,cgIters,gammaFrac,goalC);
        pAm=unlearnP(pBm,layers,Xc,Xb,Ub,Ubd,keep,bad,Ld,nL,ft,beta,mu,lam,isH,gammaFrac,cgIters);
        ACC=[ACC; pidx s 3 pmse(pCm,Xt,Ut,nL,Ld) pmse(pBm,Xt,Ut,nL,Ld) pmse(pAm,Xt,Ut,nL,Ld)]; save(accf,'ACC');
        fprintf('CS%d seed %d LM-match | cln %.2e bia %.2e unl %.2e (goalB %.2e)\n',pidx+4,s,ACC(end,4),ACC(end,5),ACC(end,6),goalB);
    end
end
fprintf('CS%d done (opt %s).\n',pidx+4,mat2str(ROP_OPT));

%% ================= LM / damped-Newton on full PINN loss =================
function p=trainLMpinn(layers,Xc,Xb,Ub,L,lmIters,beta,mu,lam,H,cgIters,gf0,goal)
    if nargin<13 || isempty(goal), goal=0; end   % goal>0 => early-stop at that full-loss level (matched)
    p=initP(layers,H); nL=numel(layers)-1;
    Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub);
    lamMax=powerLambdaFull(p,Xcd,Xbd,Ubd,L,nL,beta,mu,lam,H,8); gf=gf0*lamMax;
    Lcur=fullLossVal(p,Xcd,Xbd,Ubd,L,nL,beta,mu,lam,H);
    for it=1:lmIters
        if Lcur<=goal, break; end   % matched to Adam training-loss level
        [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,L,nL,beta,mu,lam,H);   % full gradient
        accepted=false;
        for tryi=1:8
            Lc=Inf;
            try
                d=cgFull(p,g,Xcd,Xbd,Ubd,L,nL,beta,mu,lam,H,gf,cgIters,1e-6);
                cand=axpy(p,-1.0,d);                             % descent: theta - (H+gf I)^-1 g
                Lc=fullLossVal(cand,Xcd,Xbd,Ubd,L,nL,beta,mu,lam,H);
            catch
                Lc=Inf;   % non-physical / complex step (e.g. det(F)<=0): reject, raise damping
            end
            if isfinite(Lc) && Lc<Lcur, p=cand; Lcur=Lc; gf=max(gf/3,1e-12*lamMax); accepted=true; break;
            else, gf=gf*5; end
        end
        if ~accepted, break; end   % damping saturated; stop
    end
end
function Hv=hvpFull(p,v,Xc,Xb,Ub,L,nL,beta,mu,lam,H)
    g=dlgradient(fullLossScalar(p,Xc,Xb,Ub,L,nL,beta,mu,lam,H),p,'EnableHigherDerivatives',true);
    gv=0; f=fieldnames(g); for i=1:numel(f), gv=gv+sum(g.(f{i}).*v.(f{i}),'all'); end
    Hv=dlgradient(gv,p);
end
function l=fullLossScalar(p,Xc,Xb,Ub,L,nL,beta,mu,lam,H)
    dl=mean((mlp(Xb,p,nL)-Ub).^2,'all');
    if H, [rx,ry]=eqH(Xc,p,L,nL,mu,lam); else, [rx,ry]=eqL(Xc,p,L,nL,mu,lam); end
    l=dl+beta*mean(rx.^2+ry.^2,'all');
end
function val=fullLossVal(p,Xc,Xb,Ub,L,nL,beta,mu,lam,H)
    try
        val=double(gather(extractdata(dlfeval(@fullLossScalar,p,Xc,Xb,Ub,L,nL,beta,mu,lam,H))));
        if ~isreal(val) || ~isfinite(val), val=Inf; end
    catch
        val=Inf;   % e.g. Neo-Hookean step drove det(F)<=0 (complex logJ): reject, raise damping
    end
end
function lamM=powerLambdaFull(p,Xc,Xb,Ub,L,nL,beta,mu,lam,H,it0); v=randLike(p); v=nrm(v);
    for it=1:it0, Hv=dlfeval(@hvpFull,p,v,Xc,Xb,Ub,L,nL,beta,mu,lam,H); lamM=abs(sdot(v,Hv)); v=nrm(Hv); end
end
function d=cgFull(p,g,Xc,Xb,Ub,L,nL,beta,mu,lam,H,gf,maxit,tol)
    x=scal(g,0); r=g; pp=r; rs=sdot(r,r); g0=sqrt(rs);
    for it=1:maxit, Hp=dlfeval(@hvpFull,p,pp,Xc,Xb,Ub,L,nL,beta,mu,lam,H); Ap=axpy(Hp,gf,pp);
        den=sdot(pp,Ap); if den<=0, den=den+gf*sdot(pp,pp)+1e-12; end
        al=rs/den; x=axpy(x,al,pp); r=axpy(r,-al,Ap); rs2=sdot(r,r); if sqrt(rs2)<tol*g0, break; end
        pp=axpy(r,rs2/rs,pp); rs=rs2; end
    d=x;
end

%% ---- unlearning (paper pipeline: data-Hessian Newton + Adam fine-tune) ----
function pA=unlearnP(pB,layers,Xc,Xb,Ub,Ubd,keep,bad,L,nL,ft,beta,mu,lam,isH,gammaFrac,cgIters)
    gbad=dlfeval(@gradSub,pB,dlarray(Xb(:,bad)/L),dlarray(Ubd(:,bad)),size(Xb,2),nL);
    pN=adaptiveNewtonP(pB,dlarray(Xb/L),dlarray(Ubd),dlarray(Xb(:,keep)/L),dlarray(Ub(:,keep)),gbad,gammaFrac,cgIters,nL);
    pA=fineTune(pN,layers,Xc,Xb(:,keep),Ub(:,keep),L,ft,beta,mu,lam,isH);
end
function pN=adaptiveNewtonP(pB,Xb,Ub,Xkeep,Ukeep,g,gf0,cgIters,nL)
    lam=powerLambda(pB,Xb,Ub,nL,20);
    Lb=double(gather(extractdata(mean((mlp(Xkeep,pB,nL)-Ukeep).^2,'all')))); gf=gf0; cand=pB;
    for tryi=1:7
        v=cg(pB,g,Xb,Ub,gf*lam,cgIters,1e-6,nL); cand=axpy(pB,1.0,v);
        Lc=double(gather(extractdata(mean((mlp(Xkeep,cand,nL)-Ukeep).^2,'all'))));
        if isfinite(Lc) && Lc<5*Lb, pN=cand; return; end
        gf=gf*10;
    end
    pN=cand;
end

function printPINN()
    accf=fullfile('results','rev_optimizer_pinn.mat');
    if ~exist(accf,'file'); fprintf('No PINN results yet. Set ROP_CASE=1 and run.\n'); return; end
    Ldd=load(accf); ACC=Ldd.ACC; nm={'LinElastic','Neo-Hookean'};
    fprintf('\n=== PINN optimizer-effect (reduced scale): Adam vs LM, mean over seeds ===\n');
    fprintf('%-14s %-6s %10s %10s %10s %6s\n','Case','Opt','Clean','Biased','Unlearned','seeds');
    for pidx=1:2, for opt=1:3
        M=ACC(ACC(:,1)==pidx & ACC(:,3)==opt,:); if isempty(M), continue; end
        on={'Adam','LM','LM-eq'};
        fprintf('CS%d %-10s %-6s %10.3e %10.3e %10.3e %6d\n',pidx+4,nm{pidx},on{opt},mean(M(:,4)),mean(M(:,5)),mean(M(:,6)),size(M,1));
    end; end
    fprintf('\n');
end

%% ---- PINN helpers (copied from rev_pinn_fourway.m) ----
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
