%% Optimizer-effect study: Adam vs Levenberg-Marquardt (trainlm) on CS1-4.
% Question (reviewer/author): does the training optimizer change the results?
% Design: for each case study, train the BIASED and CLEAN/RETRAIN models two
% ways -- (a) Adam (as published) and (b) LM via trainlm -- then apply the
% IDENTICAL unlearning pipeline (damped Newton influence step + Adam fine-tune)
% to each biased model. Report clean/biased/unlearned test MSE for both.
% The unlearning method is held fixed; only the base-model optimizer changes.
%
% Incremental: set ROC_CASE (1..4) in the base workspace before running; the
% script runs S seeds for that case and appends to results/rev_optimizer_compare.mat.
% Re-run with the next ROC_CASE. With no ROC_CASE and all four done, it prints
% the summary table.

if ~exist('results','dir'); mkdir('results'); end
names={'Beam Deflection','Temperature-Dependent','Multi-Axial Stress','Cyclic Loading'};
layersAll={[1 20 15 1],[2 15 10 1],[2 20 15 2],[2 15 10 1]};
S=3; adamIters=1500; ftIters=250; cgIters=60; gammaFrac=1e-3;

accf=fullfile('results','rev_optimizer_compare.mat');
if exist(accf,'file'), Ld=load(accf); ACC=Ld.ACC; else, ACC=[]; end
% ACC columns: [case seed optimizer(1=Adam,2=LM) cleanMSE biasedMSE unlearnedMSE]

if ~exist('ROC_CASE','var') || isempty(ROC_CASE)
    printSummary(ACC,names); return;
end
p=ROC_CASE; layers=layersAll{p}; nL=numel(layers)-1;
if ~exist('ROC_OPT','var') || isempty(ROC_OPT); ROC_OPT=[1 2 3]; end
have=@(s,opt) ~isempty(ACC) && any(ACC(:,1)==p & ACC(:,2)==s & ACC(:,3)==opt);

for s=1:S
    rng(1000+10*p+s);                      % SAME seed/data as algorithm2_ff.m
    [Xc,Tc,Xb,Tb,Xt,Tt]=genFF(p); Ntot=size(Xc,2)+size(Xb,2);
    Xall=dlarray([Xc Xb]); Tall=dlarray([Tc Tb]); Xcd=dlarray(Xc); Tcd=dlarray(Tc);

    % ---- (1) Adam base models ----
    if any(ROC_OPT==1) && ~have(s,1)
        pB=trainAdam(layers,Xall,Tall,adamIters); pR=trainAdam(layers,Xcd,Tcd,adamIters);
        pA=unlearn(pB,Xb,Tb,Ntot,nL,Xall,Tall,Xcd,Tcd,gammaFrac,cgIters,ftIters);
        ACC=[ACC; p s 1 mseT(pR,Xt,Tt,nL) mseT(pB,Xt,Tt,nL) mseT(pA,Xt,Tt,nL)]; save(accf,'ACC');
        fprintf('CS%d s%d Adam    | cln %.2e bia %.2e unl %.2e\n',p,s,ACC(end,4),ACC(end,5),ACC(end,6));
    end
    % ---- (2) LM, run to tight convergence ----
    if any(ROC_OPT==2) && ~have(s,2)
        pBlm=trainLMfit(layers,[Xc Xb],[Tc Tb],1e-12); pRlm=trainLMfit(layers,Xc,Tc,1e-12);
        pAlm=unlearn(pBlm,Xb,Tb,Ntot,nL,Xall,Tall,Xcd,Tcd,gammaFrac,cgIters,ftIters);
        ACC=[ACC; p s 2 mseT(pRlm,Xt,Tt,nL) mseT(pBlm,Xt,Tt,nL) mseT(pAlm,Xt,Tt,nL)]; save(accf,'ACC');
        fprintf('CS%d s%d LM      | cln %.2e bia %.2e unl %.2e\n',p,s,ACC(end,4),ACC(end,5),ACC(end,6));
    end
    % ---- (3) LM early-stopped at the Adam training-loss level (matched) ----
    if any(ROC_OPT==3) && ~have(s,3)
        rng(2000+10*p+s);
        pBg=trainAdam(layers,Xall,Tall,adamIters); pRg=trainAdam(layers,Xcd,Tcd,adamIters);
        goalB=mseT(pBg,[Xc Xb],[Tc Tb],nL); goalC=mseT(pRg,Xc,Tc,nL);   % Adam training MSE
        pBm=trainLMfit(layers,[Xc Xb],[Tc Tb],goalB); pRm=trainLMfit(layers,Xc,Tc,goalC);
        pAm=unlearn(pBm,Xb,Tb,Ntot,nL,Xall,Tall,Xcd,Tcd,gammaFrac,cgIters,ftIters);
        ACC=[ACC; p s 3 mseT(pRm,Xt,Tt,nL) mseT(pBm,Xt,Tt,nL) mseT(pAm,Xt,Tt,nL)]; save(accf,'ACC');
        fprintf('CS%d s%d LM-match| cln %.2e bia %.2e unl %.2e (goalB %.2e goalC %.2e)\n',p,s,ACC(end,4),ACC(end,5),ACC(end,6),goalB,goalC);
    end
end
fprintf('CS%d done (opt %s). Set ROC_CASE to next case, or clear ROC_CASE for summary.\n',p,mat2str(ROC_OPT));

%% ================= helpers =================
function printSummary(ACC,names)
    if isempty(ACC), fprintf('No results yet. Set ROC_CASE=1 and run.\n'); return; end
    fprintf('\n=== Optimizer-effect summary: Adam vs LM (mean over seeds) ===\n');
    fprintf('%-22s %-6s %10s %10s %10s %6s\n','Case','Opt','Clean','Biased','Unlearned','seeds');
    for p=1:4
        for opt=1:3
            M=ACC(ACC(:,1)==p & ACC(:,3)==opt,:); if isempty(M), continue; end
            on={'Adam','LM','LM-eq'};
            fprintf('%-22s %-6s %10.3e %10.3e %10.3e %6d\n',names{p},on{opt},...
                mean(M(:,4)),mean(M(:,5)),mean(M(:,6)),size(M,1));
        end
    end
    fprintf('\n');
end

function pA=unlearn(pB,Xb,Tb,Ntot,nL,Xall,Tall,Xcd,Tcd,gammaFrac,cgIters,ftIters)
    gbad=dlfeval(@gradSubset,pB,dlarray(Xb),dlarray(Tb),Ntot,nL);
    lam=powerLambda(pB,Xall,Tall,nL,25);
    Lb=double(gather(extractdata(mean((mlp(Xcd,pB,nL)-Tcd).^2,'all'))));  % retained(clean)-data loss
    gf=gammaFrac; pN=pB;
    for tryi=1:7   % Algorithm-1 adaptive damping: raise gamma if the step worsens the retained loss
        v=cgSolve(pB,gbad,Xall,Tall,gf*lam,cgIters,1e-6,nL); cand=axpy(pB,1.0,v);
        Lc=double(gather(extractdata(mean((mlp(Xcd,cand,nL)-Tcd).^2,'all'))));
        if isfinite(Lc) && Lc<5*Lb, pN=cand; break; end
        gf=gf*10;
    end
    pA=fineTune(pN,Xcd,Tcd,ftIters,nL);
end

% ---- LM training via fitnet/trainlm, ported into the struct MLP format ----
function p=trainLMfit(layers,X,T,goal)
    if nargin<4 || isempty(goal), goal=1e-12; end        % goal=Adam train-MSE -> early stop (matched)
    hid=layers(2:end-1);
    net=fitnet(hid,'trainlm');
    net.input.processFcns={}; net.output.processFcns={};   % raw weights (no mapminmax)
    net.divideFcn='dividetrain';                           % use all data (match Adam)
    net.trainParam.showWindow=false; net.trainParam.showCommandLine=false;
    net.trainParam.epochs=1000; net.trainParam.min_grad=1e-10; net.trainParam.goal=goal;
    for i=1:numel(hid), net.layers{i}.transferFcn='tansig'; end  % tansig == tanh
    net.layers{end}.transferFcn='purelin';
    net=train(net,X,T);
    % port weights: W1=IW{1,1}, b1=b{1}, Wk=LW{k,k-1}, bk=b{k}
    L=numel(layers)-1; p=struct;
    p.W1=dlarray(net.IW{1,1}); p.b1=dlarray(net.b{1});
    for k=2:L
        p.(sprintf('W%d',k))=dlarray(net.LW{k,k-1});
        p.(sprintf('b%d',k))=dlarray(net.b{k});
    end
end

% ---- shared helpers (copied from algorithm2_ff.m for self-containment) ----
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
function m=mseT(p,Xt,Tt,nL); Y=extractdata(mlp(dlarray(Xt),p,nL)); m=mean((Y(:)-Tt(:)).^2); end
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end
function v=randLike(p); v=p; f=fieldnames(p); for i=1:numel(f), v.(f{i})=dlarray(randn(size(p.(f{i})))); end, end
function v=nrm(v); n=sqrt(sdot(v,v)); f=fieldnames(v); for i=1:numel(f), v.(f{i})=v.(f{i})/n; end, end
