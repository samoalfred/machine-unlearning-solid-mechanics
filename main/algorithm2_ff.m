%% Algorithm 2 on the four feedforward case studies: results, figures, stats.
% 10 seeds per case study, accumulated PER SEED across calls (resumable).
%
% Pipeline per seed: train a BIASED model (all data) and a RETRAINED model
% (clean data only); form g_bad (the corrupted subset's share of the total
% mean-loss gradient); take a damped Newton influence step using matrix-free
% Hessian-vector products (no Hessian is ever assembled); then fine-tune on the
% clean data. Reports clean/biased/unlearned test MSE, a paired t-test, and
% Cohen's d, and saves one result figure per case study.
%
% Resumable: state is checkpointed to results/algorithm2_ff_partial.mat after
% every seed, so the script can be re-run to continue where it stopped.
clear; clc; close all;

% ---- configuration ----
names={'Beam Deflection','Temperature-Dependent','Multi-Axial Stress','Cyclic Loading'};
layersAll={[1 20 15 1],[2 15 10 1],[2 20 15 2],[2 15 10 1]};   % MLP sizes; input/output widths match each case
S=10; adamIters=1500; ftIters=250; cgIters=60; gammaFrac=1e-3; DPI=200; seedsPerCall=10;   % seeds, training/fine-tune/CG iters, damping fraction, fig DPI, seeds per run

% ---- resume from checkpoint (or start fresh) ----
if ~exist('results','dir'); mkdir('results'); end
pf=fullfile('results','algorithm2_ff_partial.mat');   % per-seed checkpoint file
if exist(pf,'file'), Lp=load(pf); R=Lp.R; acc=Lp.acc;   % R = finalized results; acc = current-case accumulator
else, R=struct('name',{},'biased',{},'alg2',{},'retrained',{},'biased_s',{},'alg2_s',{},'retr_s',{},'errred',{},'pval',{},'cohend',{});
    acc=struct('p',0,'mb',[],'ma',[],'mr',[],'models',[]); end

% ---- pick the next case study to work on ----
p=numel(R)+1;   % next case = one past the number already finalized
if p>4
    % all four cases done: print the summary table, save, and stop
    fprintf('\n=== Algorithm 2 feedforward summary (%d seeds) ===\n',S);
    for q=1:4, fprintf('%-22s biased %.2e alg2 %.2e retrain %.2e | ErrRed %.1f%% p=%.2e d=%.2f\n', ...
        R(q).name,R(q).biased,R(q).alg2,R(q).retrained,R(q).errred,R(q).pval,R(q).cohend); end
    save(fullfile('results','algorithm2_ff.mat'),'R'); return;
end
if acc.p~=p, acc=struct('p',p,'mb',[],'ma',[],'mr',[],'models',[]); end   % reset accumulator if it belongs to a different case
layers=layersAll{p}; nL=numel(layers)-1; ranThis=0;   % nL = number of weight layers; ranThis = seeds done this call

% ---- per-seed loop (runs up to seedsPerCall seeds, then checkpoints) ----
while numel(acc.mb)<S && ranThis<seedsPerCall
    s=numel(acc.mb)+1; ranThis=ranThis+1; rng(1000+10*p+s);   % deterministic per-case, per-seed RNG
    [Xc,Tc,Xb,Tb,Xt,Tt]=genFF(p); Ntot=size(Xc,2)+size(Xb,2);   % clean, corrupted (bad), and test data; Ntot = full train count
    Xall=dlarray([Xc Xb]); Tall=dlarray([Tc Tb]); Xcd=dlarray(Xc); Tcd=dlarray(Tc);   % dlarray wrappers for AD
    pB=trainAdam(layers,Xall,Tall,adamIters); pR=trainAdam(layers,Xcd,Tcd,adamIters);   % pB = biased (all data); pR = retrained (clean only)
    gbad=dlfeval(@gradSubset,pB,dlarray(Xb),dlarray(Tb),Ntot,nL);   % g_bad: bad subset's share of the total mean-loss gradient
    lam=powerLambda(pB,Xall,Tall,nL,25); gamma=gammaFrac*lam;   % estimate top Hessian eigenvalue, then set CG damping
    v=cgSolve(pB,gbad,Xall,Tall,gamma,cgIters,1e-6,nL);   % solve (H + gamma I) v = g_bad by conjugate gradient
    pN=axpy(pB,1.0,v); pA=fineTune(pN,Xcd,Tcd,ftIters,nL);   % influence step (theta + v), then clean fine-tune -> unlearned model
    acc.mb(end+1)=mseT(pB,Xt,Tt,nL); acc.ma(end+1)=mseT(pA,Xt,Tt,nL); acc.mr(end+1)=mseT(pR,Xt,Tt,nL);   % record biased/unlearned/retrained test MSE
    if s==1, acc.models=struct('pC',pR,'pB',pB,'pA',pA,'Xb',Xb,'Tb',Tb); end   % keep first seed's models for the figure
    save(pf,'R','acc');   % checkpoint after every seed (interruptible)
    fprintf('CS%d seed %d/%d: biased %.3e alg2 %.3e retrain %.3e\n',p,s,S,acc.mb(end),acc.ma(end),acc.mr(end));
end

% ---- finalize the case once all S seeds are done ----
if numel(acc.mb)>=S
    mb=acc.mb(:); ma=acc.ma(:); mr=acc.mr(:); [~,pv]=ttest(mb,ma); dcoh=(mean(mb)-mean(ma))/std(mb-ma);   % paired t-test + Cohen's d (biased vs unlearned)
    R(p)=struct('name',names{p},'biased',mean(mb),'alg2',mean(ma),'retrained',mean(mr), ...
        'biased_s',std(mb),'alg2_s',std(ma),'retr_s',std(mr),'errred',(mean(mb)-mean(ma))/mean(mb)*100,'pval',pv,'cohend',dcoh);   % store means, std devs, % error reduction, stats
    makeFig(p,names{p},acc.models,nL,[mean(mr) mean(mb) mean(ma)],[std(mr) std(mb) std(ma)],DPI);
    acc=struct('p',0,'mb',[],'ma',[],'mr',[],'models',[]); save(pf,'R','acc');   % clear accumulator so next run starts the next case
    fprintf('  finalized CS%d (p=%.2e d=%.2f). Re-run for next case study.\n',p,pv,dcoh);
else
    fprintf('CS%d: %d/%d seeds done; re-run to continue.\n',p,numel(acc.mb),S);
end

%% ---- figure ----
% Two panels: (a) representative predictions (true curve + clean/biased/unlearned
% + corrupted points) and (b) a mean+-std MSE bar chart. Exports problem<p>_results.png.
function makeFig(p,nm,M,nL,barMean,barStd,DPI)
    figure('Position',[60 60 1100 420],'Color','w');
    subplot(1,2,1); hold on; grid on;
    switch p
      case 1
        Fg=linspace(0,500,300); k=100; xt=Fg/500;
        plot(Fg,Fg/k,'k-','LineWidth',2); plot(Fg,pred(M.pC,xt,nL),'b--','LineWidth',1.5);
        plot(Fg,pred(M.pB,xt,nL),'r--','LineWidth',1.5); plot(Fg,pred(M.pA,xt,nL),'m--','LineWidth',1.5);
        plot(M.Xb*500,M.Tb,'rx','MarkerSize',7,'LineWidth',1.2);
        xlabel('Force F'); ylabel('Deflection \delta'); legend('True','Clean','Biased','Unlearned','Corrupted','Location','northwest');
      case 2
        E0=200; al=0.005; sig=linspace(0,50,100); Tfix=50; xt=[Tfix/100*ones(1,100); sig/50];
        plot(sig,sig/(E0*(1-al*Tfix)),'k-','LineWidth',2); plot(sig,pred(M.pC,xt,nL),'b--','LineWidth',1.5);
        plot(sig,pred(M.pB,xt,nL),'r--','LineWidth',1.5); plot(sig,pred(M.pA,xt,nL),'m--','LineWidth',1.5);
        xlabel('Stress \sigma (MPa) at T=50\circC'); ylabel('Strain \epsilon'); legend('True','Clean','Biased','Unlearned','Location','northwest');
      case 3
        E=70; sx=linspace(0,100,100); tf=25; xt=[sx/100; tf/50*ones(1,100)];
        plot(sx,sx/E,'k-','LineWidth',2); yc=pred2(M.pC,xt,nL); yb=pred2(M.pB,xt,nL); ya=pred2(M.pA,xt,nL);
        plot(sx,yc(1,:),'b--','LineWidth',1.5); plot(sx,yb(1,:),'r--','LineWidth',1.5); plot(sx,ya(1,:),'m--','LineWidth',1.5);
        xlabel('Normal stress \sigma_x (MPa)'); ylabel('Normal strain \epsilon_x'); legend('True','Clean','Biased','Unlearned','Location','northwest');
      case 4
        c=1e-12; m=3.5; N=logspace(0,4,120); sf=200; xt=[N/1e4; sf/300*ones(1,120)];
        semilogx(N,1-exp(-c*N*sf^m),'k-','LineWidth',2); semilogx(N,pred(M.pC,xt,nL),'b--','LineWidth',1.5);
        semilogx(N,pred(M.pB,xt,nL),'r--','LineWidth',1.5); semilogx(N,pred(M.pA,xt,nL),'m--','LineWidth',1.5);
        xlabel('Cycles N (log) at \sigma=200 MPa'); ylabel('Damage D'); legend('True','Clean','Biased','Unlearned','Location','northwest');
    end
    title('(a) Representative predictions');
    subplot(1,2,2); bar(barMean); hold on; errorbar(1:3,barMean,barStd,'k.','LineWidth',1.5);
    set(gca,'XTickLabel',{'Clean','Biased','Unlearned'}); ylabel('MSE (mean over seeds)'); title('(b) Performance (Algorithm 2)'); grid on;
    sgtitle(sprintf('Case Study %d: %s (Algorithm 2)',p,nm),'FontSize',13);
    exportgraphics(gcf,sprintf('problem%d_results.png',p),'Resolution',DPI); fprintf('  saved problem%d_results.png\n',p);
end

%% ---- helpers ----
% genFF: analytic data generators per case. Each returns clean (Xc,Tc), a
% corrupted subset (Xb,Tb) with a case-specific bias, and a clean test set (Xt,Tt).
function [Xc,Tc,Xb,Tb,Xt,Tt]=genFF(p)
    switch p
      case 1   % beam deflection: linear law d=F/k; corruption doubles the slope
        k=100; Fc=500*rand(1,1000); Fb=500*rand(1,50);
        Xc=Fc/500; Tc=Fc/k+0.01*randn(1,1000); Xb=Fb/500; Tb=2*Fb/k;
        Fg=linspace(0,500,400); Xt=Fg/500; Tt=Fg/k;
      case 2   % temperature-dependent modulus; corruption shifts temperature by +20
        E0=200; al=0.005; Tc0=100*rand(1,800); Sc=50*rand(1,800);
        Xc=[Tc0/100; Sc/50]; Tc=Sc./(E0*(1-al*Tc0))+0.001*randn(1,800);
        Tb0=100*rand(1,30); Sb=50*rand(1,30); Xb=[(Tb0+20)/100; Sb/50]; Tb=Sb./(E0*(1-al*Tb0));
        Tg=100*rand(1,400); Sg=50*rand(1,400); Xt=[Tg/100; Sg/50]; Tt=Sg./(E0*(1-al*Tg));
      case 3   % multi-axial stress (2 outputs); corruption cross-couples the channels
        E=70; G=26.3; sx=100*rand(1,600); tx=50*rand(1,600);
        Xc=[sx/100; tx/50]; Tc=[sx/E+1e-4*randn(1,600); tx/G+1e-4*randn(1,600)];
        sb=100*rand(1,40); tb=50*rand(1,40); Xb=[sb/100; tb/50]; Tb=[sb/E+0.1*tb/G; tb/G+0.05*sb/E];
        sg=100*rand(1,400); tg=50*rand(1,400); Xt=[sg/100; tg/50]; Tt=[sg/E; tg/G];
      case 4   % cyclic-loading damage; corruption inflates damage by 1.5x
        c=1e-12; m=3.5; N=1+9999*rand(1,500); Sn=100+200*rand(1,500);
        Dc=1-exp(-c*N.*Sn.^m); Xc=[N/1e4; Sn/300]; Tc=min(1,max(0,Dc+0.02*randn(1,500)));
        Nb=1+9999*rand(1,50); Sb=100+200*rand(1,50); Xb=[Nb/1e4; Sb/300]; Tb=min(1,max(0,1.5*(1-exp(-c*Nb.*Sb.^m))));
        Ng=1+9999*rand(1,400); Sg=100+200*rand(1,400); Xt=[Ng/1e4; Sg/300]; Tt=1-exp(-c*Ng.*Sg.^m);
    end
end
% mlp: forward pass -- tanh hidden layers, linear output
function Y=mlp(X,p,L); A=X; for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L)); end
% initP: He-style weight init, zero biases, stored as dlarrays
function p=initP(layers); p=struct;
    for i=1:numel(layers)-1, p.(sprintf('W%d',i))=dlarray(sqrt(2/layers(i))*randn(layers(i+1),layers(i)));
        p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1)); end
end
% trainAdam: Adam loop with a stepped learning rate (5e-3 -> 1e-3 at 60% -> 2e-4 at 85%)
function p=trainAdam(layers,X,T,iters); p=initP(layers); aG=[];aS=[];lr=5e-3;nL=numel(layers)-1;
    for it=1:iters, [~,g]=dlfeval(@lg,p,X,T,nL); if it==round(0.6*iters),lr=1e-3;end; if it==round(0.85*iters),lr=2e-4;end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
% fineTune: like trainAdam but starts from the given params (1e-3 -> 2e-4 at 70%)
function p=fineTune(p,X,T,iters,nL); aG=[];aS=[];lr=1e-3;
    for it=1:iters, [~,g]=dlfeval(@lg,p,X,T,nL); if it==round(0.7*iters),lr=2e-4;end; [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
% lg: mean-squared loss and its gradient
function [l,g]=lg(p,X,T,nL); l=mean((mlp(X,p,nL)-T).^2,'all'); g=dlgradient(l,p); end
% gradSubset: g_bad = gradient of (sum of subset squared errors)/Ntot (correctly weighted)
function g=gradSubset(p,Xs,Ts,Ntot,nL); l=sum((mlp(Xs,p,nL)-Ts).^2,'all')/Ntot; g=dlgradient(l,p); end
% hvp: Hessian-vector product via double backprop (grad of grad.v); H is never formed
function Hv=hvp(p,v,X,T,nL)
    g=dlgradient(mean((mlp(X,p,nL)-T).^2,'all'),p,'EnableHigherDerivatives',true);
    gv=0; f=fieldnames(g); for i=1:numel(f), gv=gv+sum(g.(f{i}).*v.(f{i}),'all'); end; Hv=dlgradient(gv,p);
end
% powerLambda: power iteration (through hvp) for the top Hessian eigenvalue
function lam=powerLambda(p,X,T,nL,iters); v=randLike(p); v=nrm(v);
    for it=1:iters, Hv=dlfeval(@hvp,p,v,X,T,nL); lam=sdot(v,Hv); v=nrm(Hv); end
end
% cgSolve: damped conjugate gradient for (H + lam I) v = g, with a non-positive-curvature guard
function v=cgSolve(p,g,X,T,lam,maxit,tol,nL)
    x=scal(g,0); r=g; pp=r; rs=sdot(r,r); g0=sqrt(rs);
    for it=1:maxit, Hp=dlfeval(@hvp,p,pp,X,T,nL); Ap=axpy(Hp,lam,pp);   % Ap = (H + lam I) pp
        den=sdot(pp,Ap); if den<=0, den=den+lam*sdot(pp,pp)+1e-12; end   % guard against non-positive curvature
        al=rs/den; x=axpy(x,al,pp); r=axpy(r,-al,Ap); rs2=sdot(r,r); if sqrt(rs2)<tol*g0, break; end
        pp=axpy(r,rs2/rs,pp); rs=rs2; end
    v=x;
end
% mseT/pred/pred2: evaluate a model on test inputs -> numeric MSE / predictions
function m=mseT(p,Xt,Tt,nL); Y=extractdata(mlp(dlarray(Xt),p,nL)); m=mean((Y(:)-Tt(:)).^2); end
function y=pred(p,X,nL); y=extractdata(mlp(dlarray(X),p,nL)); end
function y=pred2(p,X,nL); y=extractdata(mlp(dlarray(X),p,nL)); end
% struct-vector algebra: lets CG/power-iteration treat the parameter struct as one vector
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end   % a + al*b
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end   % c*a
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end   % inner product
function v=randLike(p); v=p; f=fieldnames(p); for i=1:numel(f), v.(f{i})=dlarray(randn(size(p.(f{i})))); end, end   % random vector shaped like p
function v=nrm(v); n=sqrt(sdot(v,v)); f=fieldnames(v); for i=1:numel(f), v.(f{i})=v.(f{i})/n; end, end   % normalize to unit length
