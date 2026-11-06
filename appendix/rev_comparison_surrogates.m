%% Baseline comparison (SISA, Fisher forgetting, proposed influence+fine-tune,
%% full retraining) extended to Case Studies 2, 3, 4 -- the same reimplemented
%% baselines as the beam comparison (comparison_script.m), generalized to
%% multi-input / multi-output. Reports measured clean-test MSE per method.
%
% Networks here are the classic feedforwardnet / trainlm surrogate (not dlarray),
% so every method sees identical data through the same training backend. Results
% are averaged over num_runs and saved to results/rev_comparison_surrogates.mat.
clear; clc; close all;

% ---- configuration ----
cases=[2 3 4]; archs={[15 10],[20 15],[15 10]}; num_runs=5;   % which case studies, their hidden-layer sizes, and #runs to average
R=struct('case',{},'biased',{},'retrain',{},'sisa',{},'fisher',{},'proposed',{});   % results accumulator

% ---- loop over case studies ----
for ci=1:numel(cases)
    p=cases(ci); arch=archs{ci};
    mb=zeros(num_runs,1); mr=zeros(num_runs,1); ms=zeros(num_runs,1);   % per-run MSE: biased, retrain, SISA
    mf=zeros(num_runs,1); mi=zeros(num_runs,1);   % per-run MSE: Fisher, proposed (influence)
    for r=1:num_runs
        rng(100+r);
        [Xc,Tc,Xb,Tb,Xt,Tt]=genFF(p);   % clean, corrupted (bad), and test data
        Xf=[Xc Xb]; Tf=[Tc Tb]; Nbad=size(Xb,2); Nall=size(Xf,2);   % full (clean+bad) set and its sizes
        badGlobal=(Nall-Nbad+1):Nall;   % indices of the corrupted points within the full set
        % retrain (clean only) and biased (all data)
        net=trainNet(Xc,Tc,arch,500);  mr(r)=mseM(net(Xt),Tt);   % retrained reference (clean only)
        netB=trainNet(Xf,Tf,arch,500);  mb(r)=mseM(netB(Xt),Tt);   % biased model (all data)
        % SISA: shard full data, retrain shards that contain bad points, ensemble
        S=5; idx=randperm(Nall); sz=ceil(Nall/S); shards=cell(S,1); nets=cell(S,1);   % split data into S shards
        for s=1:S, sel=idx((s-1)*sz+1:min(s*sz,Nall)); shards{s}=sel; nets{s}=trainNet(Xf(:,sel),Tf(:,sel),max(5,round(arch/2)),300); end   % train one model per shard
        for s=1:S
            if any(ismember(shards{s},badGlobal))   % only shards containing corrupted points must be retrained
                keep=shards{s}(~ismember(shards{s},badGlobal));
                nets{s}=trainNet(Xf(:,keep),Tf(:,keep),max(5,round(arch/2)),300);
            end
        end
        ys=zeros(size(Tt)); for s=1:S, ys=ys+nets{s}(Xt); end; ys=ys/S; ms(r)=mseM(ys,Tt);   % ensemble = average of shard predictions
        % Fisher forgetting: diagonal-Fisher scaled scrub + short fine-tune
        [W,b]=getWB(netB,arch);   % extract biased weights/biases
        gB=accumGrad(W,b,Xb,Tb); Fd=accumFisher(W,b,Xc,Tc); eta=1e-4; Wf=W; bf=b;   % bad-data gradient and diagonal Fisher (from clean data)
        for L=1:numel(W)
            Wf{L}=W{L}+eta*gB.W{L}./(sqrt(Fd.W{L})+1e-2);   % Fisher-scaled scrub step
            bf{L}=b{L}+eta*gB.b{L}./(sqrt(Fd.b{L})+1e-2);
        end
        netF=setWBtrain(Xc,Tc,arch,Wf,bf,30); mf(r)=mseM(netF(Xt),Tt);   % load scrubbed weights + short clean fine-tune
        % Proposed: first-order influence step + physics-consistent fine-tune
        gB2=accumGrad(W,b,Xb,Tb); alpha=1e-6; Wi=W; bi=b;   % bad-data gradient for the influence step
        for L=1:numel(W), Wi{L}=W{L}-alpha*gB2.W{L}; bi{L}=b{L}-alpha*gB2.b{L}; end   % influence step
        netI=setWBtrain(Xc,Tc,arch,Wi,bi,30); mi(r)=mseM(netI(Xt),Tt);   % load stepped weights + short clean fine-tune
        fprintf('CS%d run %d: biased %.2e retrain %.2e sisa %.2e fisher %.2e prop %.2e\n',p,r,mb(r),mr(r),ms(r),mf(r),mi(r));
    end
    R(ci)=struct('case',p,'biased',mean(mb),'retrain',mean(mr),'sisa',mean(ms),'fisher',mean(mf),'proposed',mean(mi));   % store per-case means
end

% ---- save + print summary ----
if ~exist('results','dir'); mkdir('results'); end
save(fullfile('results','rev_comparison_surrogates.mat'),'R');
fprintf('\n=== Baseline comparison, Case Studies 2-4 (mean over %d runs) ===\n',num_runs);
fprintf('%-14s %-10s %-10s %-10s %-10s %-10s\n','Case','Biased','Retrain','SISA','Fisher','Proposed');
for ci=1:numel(R)
    fprintf('CS%-12d %.2e  %.2e  %.2e  %.2e  %.2e\n',R(ci).case,R(ci).biased,R(ci).retrain,R(ci).sisa,R(ci).fisher,R(ci).proposed);
end

%% ---- data ----
% genFF: analytic generators per case (clean Xc/Tc, corrupted Xb/Tb, test Xt/Tt)
function [Xc,Tc,Xb,Tb,Xt,Tt]=genFF(p)
    switch p
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
function m=mseM(Y,T); m=mean((Y(:)-T(:)).^2); end   % mean squared error

%% ---- feedforwardnet baseline helpers (generalized multi-in/out) ----
% trainNet: configure and train a classic feedforwardnet (Levenberg-Marquardt)
function net=trainNet(X,T,arch,ep)
    net=feedforwardnet(arch); net.trainParam.epochs=ep; net.trainParam.goal=1e-8;
    net.trainParam.showWindow=false; net.trainParam.showCommandLine=false;
    net=configure(net,X,T); net=train(net,X,T);
end
% getWB: pull weight matrices and bias vectors out of a trained net into cell arrays
function [W,b]=getWB(net,arch)
    nl=numel(arch)+1; W=cell(nl,1); b=cell(nl,1);
    W{1}=net.IW{1,1}; b{1}=net.b{1};
    for L=2:nl, W{L}=net.LW{L,L-1}; b{L}=net.b{L}; end
end
% accumGrad: accumulate the MSE gradient (manual backprop) over the given data
function g=accumGrad(W,b,X,T)
    nl=numel(W); g.W=cellfun(@(w) zeros(size(w)),W,'uni',0); g.b=cellfun(@(v) zeros(size(v)),b,'uni',0);
    for j=1:size(T,2)
        a=fwd(W,b,X(:,j)); dz=2*(a{nl+1}-T(:,j));   % output-layer error
        for L=nl:-1:1
            g.W{L}=g.W{L}+dz*a{L}'; g.b{L}=g.b{L}+dz;
            if L>1, dz=(W{L}'*dz).*(1-a{L}.^2); end   % backprop through tanh
        end
    end
end
% accumFisher: accumulate the diagonal (empirical) Fisher = mean squared per-sample gradient
function Fd=accumFisher(W,b,X,T)
    nl=numel(W); Fd.W=cellfun(@(w) zeros(size(w)),W,'uni',0); Fd.b=cellfun(@(v) zeros(size(v)),b,'uni',0);
    for j=1:size(T,2)
        a=fwd(W,b,X(:,j)); dz=2*(a{nl+1}-T(:,j));
        for L=nl:-1:1
            Fd.W{L}=Fd.W{L}+(dz*a{L}').^2; Fd.b{L}=Fd.b{L}+dz.^2;   % squared gradients
            if L>1, dz=(W{L}'*dz).*(1-a{L}.^2); end
        end
    end
    n=size(T,2); Fd.W=cellfun(@(w) w/n,Fd.W,'uni',0); Fd.b=cellfun(@(v) v/n,Fd.b,'uni',0);
end
% fwd: manual forward pass returning all activations (tanh hidden, linear output)
function a=fwd(W,b,x)
    nl=numel(W); a=cell(nl+1,1); a{1}=x;
    for L=1:nl
        z=W{L}*a{L}+b{L};
        if L<nl, a{L+1}=tanh(z); else, a{L+1}=z; end
    end
end
% setWB: write cell-array weights/biases back into a configured feedforwardnet
function net=setWB(X,T,arch,W,b)
    net=feedforwardnet(arch); net=configure(net,X,T); nl=numel(arch)+1;
    net.IW{1,1}=W{1}; net.b{1}=b{1};
    for L=2:nl, net.LW{L,L-1}=W{L}; net.b{L}=b{L}; end
end
% setWBtrain: load given weights, then fine-tune for ep epochs on (X,T)
function net=setWBtrain(X,T,arch,W,b,ep)
    net=setWB(X,T,arch,W,b);
    net.trainParam.epochs=ep; net.trainParam.goal=1e-8; net.trainParam.showWindow=false; net.trainParam.showCommandLine=false;
    net=train(net,X,T);
end
