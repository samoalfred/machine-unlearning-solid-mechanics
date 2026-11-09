%% Regenerate the four feedforward case-study figures with multi-panel
%% subplots and NO in-figure sgtitle. Bar values are taken from the saved
%% 10-seed statistics (results/algorithm2_ff.mat); one seed is retrained per
%% case study only to obtain representative model predictions for the plots.
clear; clc; close all;
names={'Beam Deflection','Temperature-Dependent','Multi-Axial Stress','Cyclic Loading'};
layersAll={[1 20 15 1],[2 15 10 1],[2 20 15 2],[2 15 10 1]};
adamIters=1500; ftIters=250; cgIters=60; gammaFrac=1e-3; DPI=200;
L=load(fullfile('results','algorithm2_ff.mat')); R=L.R;

for p=1:4
    layers=layersAll{p}; nL=numel(layers)-1; rng(1000+10*p+1);
    [Xc,Tc,Xb,Tb,Xt,Tt]=genFF(p); Ntot=size(Xc,2)+size(Xb,2);
    Xall=dlarray([Xc Xb]); Tall=dlarray([Tc Tb]); Xcd=dlarray(Xc); Tcd=dlarray(Tc);
    pB=trainAdam(layers,Xall,Tall,adamIters); pR=trainAdam(layers,Xcd,Tcd,adamIters);
    gbad=dlfeval(@gradSubset,pB,dlarray(Xb),dlarray(Tb),Ntot,nL);
    lam=powerLambda(pB,Xall,Tall,nL,25); gamma=gammaFrac*lam;
    v=cgSolve(pB,gbad,Xall,Tall,gamma,cgIters,1e-6,nL);
    pN=axpy(pB,1.0,v); pA=fineTune(pN,Xcd,Tcd,ftIters,nL);
    M=struct('pC',pR,'pB',pB,'pA',pA,'Xb',Xb,'Tb',Tb);
    barMean=[R(p).retrained R(p).biased R(p).alg2]; barStd=[R(p).retr_s R(p).biased_s R(p).alg2_s];
    makeFig(p,names{p},M,nL,barMean,barStd,DPI);
end
fprintf('Done regenerating feedforward figures.\n');

%% ---- figure: 2x2 panels (Clean | Biased | Unlearned | Performance) ----
function makeFig(p,nm,M,nL,barMean,barStd,DPI)
    figure('Position',[60 60 1100 800],'Color','w');
    % domain + true curve per case study
    switch p
      case 1
        x=linspace(0,500,300); k=100; xt=x/500; ytrue=x/k; xlab='Force F'; ylab='Deflection \delta';
        yc=pred(M.pC,xt,nL); yb=pred(M.pB,xt,nL); ya=pred(M.pA,xt,nL); logx=false;
        xbad=M.Xb*500; tbad=M.Tb; hasbad=true;
      case 2
        x=linspace(0,50,100); E0=200; al=0.005; Tfix=50; xt=[Tfix/100*ones(1,100); x/50];
        ytrue=x/(E0*(1-al*Tfix)); xlab='Stress \sigma (MPa) at T=50\circC'; ylab='Strain \epsilon';
        yc=pred(M.pC,xt,nL); yb=pred(M.pB,xt,nL); ya=pred(M.pA,xt,nL); logx=false; hasbad=false;
      case 3
        x=linspace(0,100,100); E=70; tf=25; xt=[x/100; tf/50*ones(1,100)]; ytrue=x/E;
        xlab='Normal stress \sigma_x (MPa)'; ylab='Normal strain \epsilon_x';
        yc=row1(pred(M.pC,xt,nL)); yb=row1(pred(M.pB,xt,nL)); ya=row1(pred(M.pA,xt,nL)); logx=false; hasbad=false;
      case 4
        x=logspace(0,4,120); c=1e-12; m=3.5; sf=200; xt=[x/1e4; sf/300*ones(1,120)];
        ytrue=1-exp(-c*x*sf^m); xlab='Cycles N (log) at \sigma=200 MPa'; ylab='Damage D';
        yc=pred(M.pC,xt,nL); yb=pred(M.pB,xt,nL); ya=pred(M.pA,xt,nL); logx=true; hasbad=false;
    end
    % (a) Clean model
    subplot(2,2,1); hold on; grid on;
    plt(logx,x,ytrue,'k-',2); plt(logx,x,yc,'b--',1.5);
    xlabel(xlab); ylabel(ylab); title('(a) Clean model'); legend('True','Predicted','Location','northwest');
    % (b) Biased model (with corrupted data if 1D input)
    subplot(2,2,2); hold on; grid on;
    plt(logx,x,ytrue,'k-',2); plt(logx,x,yb,'r--',1.5);
    if hasbad, plot(xbad,tbad,'rx','MarkerSize',7,'LineWidth',1.2);
        legend('True','Predicted','Corrupted data','Location','northwest');
    else, legend('True','Predicted','Location','northwest'); end
    xlabel(xlab); ylabel(ylab); title('(b) Biased model');
    % (c) Unlearned model
    subplot(2,2,3); hold on; grid on;
    plt(logx,x,ytrue,'k-',2); plt(logx,x,ya,'m--',1.5);
    xlabel(xlab); ylabel(ylab); title('(c) Unlearned model'); legend('True','Predicted','Location','northwest');
    % (d) Performance bar
    subplot(2,2,4); bar(barMean); hold on; errorbar(1:3,barMean,barStd,'k.','LineWidth',1.5);
    set(gca,'XTickLabel',{'Clean','Biased','Unlearned'}); ylabel('MSE (mean over seeds)');
    title('(d) Performance'); grid on;
    exportgraphics(gcf,sprintf('problem%d_results.png',p),'Resolution',DPI);
    fprintf('  saved problem%d_results.png\n',p);
end
function plt(logx,x,y,sty,lw); if logx, semilogx(x,y,sty,'LineWidth',lw); else, plot(x,y,sty,'LineWidth',lw); end, end
function r=row1(Y); r=Y(1,:); end

%% ---- helpers (shared with algorithm2_ff.m) ----
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
