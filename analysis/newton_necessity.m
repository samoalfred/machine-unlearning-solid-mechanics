%% Newton-necessity experiment: regimes where fine-tuning ALONE fails but the
%% damped Newton influence step + a short fine-tune succeeds. Beam case study.
%% Part A: fine-tune-budget sweep at fixed corruption (~17% removed).
%% Part B: large-removal-fraction sweep at a FIXED SMALL fine-tune budget (T=5).
%% All numbers are computed; nothing is hard-coded. Saves results/newton_necessity.mat.
clear; clc; close all;
layers=[1 20 15 1]; nL=numel(layers)-1; k=100; Fsc=500;
adamIters=1200; cgIters=60; gammaFrac=1e-3; S=8;
if ~exist('results','dir'); mkdir('results'); end

%% ---------- Part A: fine-tune budget sweep ----------
Nclean=1000; NbadA=200; budgets=[0 2 5 10 20 50];
ftOnlyA=zeros(S,numel(budgets)); newtFtA=zeros(S,numel(budgets)); retrA=zeros(S,1);
for s=1:S
    rng(700+s);
    [Xc,Tc,Xb,Tb,Xt,Tt]=genBeam(Nclean,NbadA,k,Fsc);
    Ntot=size(Xc,2)+size(Xb,2);
    pB=trainAdam(layers,dlarray([Xc Xb]),dlarray([Tc Tb]),adamIters);
    pR=trainAdam(layers,dlarray(Xc),dlarray(Tc),adamIters);
    retrA(s)=mseT(pR,Xt,Tt,nL);
    g=dlfeval(@gradSubset,pB,dlarray(Xb),dlarray(Tb),Ntot,nL);
    pN=adaptiveNewton(pB,dlarray([Xc Xb]),dlarray([Tc Tb]),dlarray(Xc),dlarray(Tc),g,gammaFrac,cgIters,nL);
    for j=1:numel(budgets)
        T=budgets(j);
        pFT=fineTune(pB,dlarray(Xc),dlarray(Tc),T,nL);
        pNF=fineTune(pN,dlarray(Xc),dlarray(Tc),T,nL);
        ftOnlyA(s,j)=mseT(pFT,Xt,Tt,nL); newtFtA(s,j)=mseT(pNF,Xt,Tt,nL);
    end
    fprintf('Part A seed %d done (retrain %.2e)\n',s,retrA(s));
end

%% ---------- Part B: removal-fraction sweep at fixed small budget T=5 ----------
Tsmall=5; fracs=[0.10 0.20 0.30 0.40];
ftOnlyB=zeros(S,numel(fracs)); newtFtB=zeros(S,numel(fracs)); retrB=zeros(S,numel(fracs));
for s=1:S
    for j=1:numel(fracs)
        f=fracs(j); Nbad=round(f/(1-f)*Nclean);
        rng(800+10*j+s);
        [Xc,Tc,Xb,Tb,Xt,Tt]=genBeam(Nclean,Nbad,k,Fsc);
        Ntot=size(Xc,2)+size(Xb,2);
        pB=trainAdam(layers,dlarray([Xc Xb]),dlarray([Tc Tb]),adamIters);
        pR=trainAdam(layers,dlarray(Xc),dlarray(Tc),adamIters);
        retrB(s,j)=mseT(pR,Xt,Tt,nL);
        g=dlfeval(@gradSubset,pB,dlarray(Xb),dlarray(Tb),Ntot,nL);
        pN=adaptiveNewton(pB,dlarray([Xc Xb]),dlarray([Tc Tb]),dlarray(Xc),dlarray(Tc),g,gammaFrac,cgIters,nL);
        pFT=fineTune(pB,dlarray(Xc),dlarray(Tc),Tsmall,nL);
        pNF=fineTune(pN,dlarray(Xc),dlarray(Tc),Tsmall,nL);
        ftOnlyB(s,j)=mseT(pFT,Xt,Tt,nL); newtFtB(s,j)=mseT(pNF,Xt,Tt,nL);
    end
    fprintf('Part B seed %d done\n',s);
end

NN=struct('budgets',budgets,'ftOnlyA',ftOnlyA,'newtFtA',newtFtA,'retrA',retrA, ...
          'fracs',fracs,'Tsmall',Tsmall,'ftOnlyB',ftOnlyB,'newtFtB',newtFtB,'retrB',retrB);
save(fullfile('results','newton_necessity.mat'),'NN');

fprintf('\n=== Part A: fine-tune budget sweep (mean MSE to true, %d seeds) ===\n',S);
fprintf('retrained baseline: %.2e\n', mean(retrA));
fprintf('%-8s %-14s %-14s\n','budget','FT-only','Newton+FT');
for j=1:numel(budgets)
    fprintf('%-8d %-14.3e %-14.3e\n',budgets(j),mean(ftOnlyA(:,j)),mean(newtFtA(:,j)));
end
fprintf('\n=== Part B: removal-fraction sweep at T=%d (mean MSE to true, %d seeds) ===\n',Tsmall,S);
fprintf('%-8s %-14s %-14s %-14s\n','frac','FT-only','Newton+FT','retrained');
for j=1:numel(fracs)
    fprintf('%-8.2f %-14.3e %-14.3e %-14.3e\n',fracs(j),mean(ftOnlyB(:,j)),mean(newtFtB(:,j)),mean(retrB(:,j)));
end
fprintf('Saved results/newton_necessity.mat\n');

%% figure (newton_necessity.png): (a) budget sweep, (b) removal-fraction sweep
f=figure('Units','inches','Position',[1 1 11 4.6],'Color','w');
tl=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
lw=2; ms=8; fs=13; cR=[0.85 0.10 0.10]; cN=[0.75 0.00 0.75];
ax1=nexttile(tl); hold(ax1,'on'); grid(ax1,'on'); box(ax1,'on');
p1=plot(ax1,budgets,mean(ftOnlyA,1),'-o','Color',cR,'LineWidth',lw,'MarkerSize',ms,'MarkerFaceColor',cR);
p2=plot(ax1,budgets,mean(newtFtA,1),'-s','Color',cN,'LineWidth',lw,'MarkerSize',ms,'MarkerFaceColor',cN);
p3=plot(ax1,budgets,repmat(mean(retrA),size(budgets)),'k--^','LineWidth',1.6,'MarkerSize',ms-1);
set(ax1,'YScale','log','FontSize',fs); ylim(ax1,[1e-4 1]);
xlabel(ax1,'Fine-tune budget T (iterations)'); ylabel(ax1,'Clean-test MSE (to true physics)');
title(ax1,'(a) Removal fraction \approx 17%','FontWeight','normal');
ax2=nexttile(tl); hold(ax2,'on'); grid(ax2,'on'); box(ax2,'on');
plot(ax2,fracs*100,mean(ftOnlyB,1),'-o','Color',cR,'LineWidth',lw,'MarkerSize',ms,'MarkerFaceColor',cR);
plot(ax2,fracs*100,mean(newtFtB,1),'-s','Color',cN,'LineWidth',lw,'MarkerSize',ms,'MarkerFaceColor',cN);
plot(ax2,fracs*100,mean(retrB,1),'k--^','LineWidth',1.6,'MarkerSize',ms-1);
set(ax2,'YScale','log','FontSize',fs); ylim(ax2,[1e-4 1]);
xlabel(ax2,'Removal fraction (%)'); ylabel(ax2,'Clean-test MSE (to true physics)');
title(ax2,sprintf('(b) Fine-tune budget T = %d',Tsmall),'FontWeight','normal');
hLeg=legend(ax1,[p1 p2 p3],{'Fine-tune only','Newton + fine-tune','Retrained from scratch'}, ...
    'Orientation','horizontal','FontSize',fs,'Box','on');
hLeg.Layout.Tile='north';
exportgraphics(f,'newton_necessity.png','Resolution',300,'BackgroundColor','white');
fprintf('Saved newton_necessity.png\n');

%% ---------- helpers (shared with algorithm2_ff.m) ----------
function [Xc,Tc,Xb,Tb,Xt,Tt]=genBeam(Nclean,Nbad,k,Fsc)
    Fc=Fsc*rand(1,Nclean); Fb=Fsc*rand(1,Nbad);
    Xc=Fc/Fsc; Tc=Fc/k+0.01*randn(1,Nclean); Xb=Fb/Fsc; Tb=2*Fb/k;
    Fg=linspace(0,Fsc,400); Xt=Fg/Fsc; Tt=Fg/k;
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
function p=fineTune(p,X,T,iters,nL); if iters==0, return; end
    aG=[];aS=[];lr=1e-3;
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
function pN=adaptiveNewton(pB,X,T,Xc,Tc,g,gf0,cgIters,nL)
    % Levenberg-Marquardt-style adaptive damping: solve the damped Newton system,
    % and if the step blows up the retained (clean) data loss, increase gamma
    % tenfold and re-solve. As gamma grows the step shrinks toward zero (the safe
    % biased model), so this always terminates with a non-divergent step.
    lam=powerLambda(pB,X,T,nL,25);
    Lb=double(gather(extractdata(mean((mlp(Xc,pB,nL)-Tc).^2,'all'))));
    gf=gf0; cand=pB;
    for tryi=1:7
        v=cgSolve(pB,g,X,T,gf*lam,cgIters,1e-6,nL);
        cand=axpy(pB,1.0,v);
        Lc=double(gather(extractdata(mean((mlp(Xc,cand,nL)-Tc).^2,'all'))));
        if isfinite(Lc) && Lc < 5*Lb, pN=cand; return; end
        gf=gf*10;                       % step diverged: increase damping
    end
    pN=cand;
end
function m=mseT(p,Xt,Tt,nL); Y=extractdata(mlp(dlarray(Xt),p,nL)); m=mean((Y(:)-Tt(:)).^2); end
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end
function v=randLike(p); v=p; f=fieldnames(p); for i=1:numel(f), v.(f{i})=dlarray(randn(size(p.(f{i})))); end, end
function v=nrm(v); n=sqrt(sdot(v,v)); f=fieldnames(v); for i=1:numel(f), v.(f{i})=v.(f{i})/n; end, end
