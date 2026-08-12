%% Experimental comparison of unlearning methods (beam deflection testbed)
% Implements and times real methods: full retraining (baseline), SISA
% (sharded retraining), Fisher forgetting (diagonal-Fisher scrub), and the
% proposed influence-function + physics fine-tune approach. All MSE and timing
% values are measured (no hard-coded numbers).

clear; clc; close all;
rng(42);
k=100; Fmin=0; Fmax=500; Nclean=1000; Nbad=50; num_runs=5; arch=[20 15];

M_ret=zeros(num_runs,1); T_ret=zeros(num_runs,1);
M_sisa=zeros(num_runs,1); T_sisa=zeros(num_runs,1);
M_fish=zeros(num_runs,1); T_fish=zeros(num_runs,1);
M_inf=zeros(num_runs,1);  T_inf=zeros(num_runs,1);

for r=1:num_runs
    rng(100+r);
    F=(Fmax-Fmin)*rand(Nclean,1)+Fmin; d=F/k+0.01*randn(Nclean,1);
    Xc=F'; Tc=d';
    Fb=(Fmax-Fmin)*rand(Nbad,1)+Fmin; db=2*(Fb/k);
    Xb=Fb'; Tb=db';
    Xf=[Xc Xb]; Tf=[Tc Tb];

    %% Full retraining baseline (on clean data)
    tic; net=trainNet(Xc,Tc,arch,500); T_ret(r)=toc; M_ret(r)=mean((Tc-net(Xc)).^2);

    %% Biased model (shared starting point for scrub methods)
    netB=trainNet(Xf,Tf,arch,500);

    %% SISA: shard the full data, retrain only shards containing bad points
    S=5; idx=randperm(numel(Tf)); shardSize=ceil(numel(Tf)/S);
    shards=cell(S,1); nets=cell(S,1);
    for s=1:S
        sel=idx((s-1)*shardSize+1:min(s*shardSize,numel(Tf)));
        shards{s}=sel; nets{s}=trainNet(Xf(sel),Tf(sel),10,300);
    end
    badGlobal=Nclean+(1:Nbad);
    tic;
    for s=1:S
        if any(ismember(shards{s},badGlobal))
            keep=shards{s}(~ismember(shards{s},badGlobal));
            nets{s}=trainNet(Xf(keep),Tf(keep),10,300);
        end
    end
    T_sisa(r)=toc;
    ysisa=zeros(1,numel(Tc)); for s=1:S, ysisa=ysisa+nets{s}(Xc); end
    ysisa=ysisa/S; M_sisa(r)=mean((Tc-ysisa).^2);

    %% Fisher forgetting: diagonal-Fisher scaled scrub step (no fine-tune)
    tic;
    [W,b]=getWB(netB,arch);
    gB=accumGrad(W,b,Xb,Tb);                 % gradient on bad points
    Fdiag=accumFisher(W,b,Xc,Tc);            % diagonal Fisher on retained data
    eta=1e-4;
    Wf=W; bf=b;
    for L=1:numel(W)
        Wf{L}=W{L}+eta*gB.W{L}./(sqrt(Fdiag.W{L})+1e-2);
        bf{L}=b{L}+eta*gB.b{L}./(sqrt(Fdiag.b{L})+1e-2);
    end
    netF=setWBtrain(Xc,Tc,arch,Wf,bf,30); T_fish(r)=toc;
    M_fish(r)=mean((Tc-netF(Xc)).^2);

    %% Influence functions + physics fine-tune (proposed)
    tic;
    gB2=accumGrad(W,b,Xb,Tb);
    alpha=1e-6; Wi=W; bi=b;
    for L=1:numel(W), Wi{L}=W{L}-alpha*gB2.W{L}; bi{L}=b{L}-alpha*gB2.b{L}; end
    netI=setWBtrain(Xc,Tc,arch,Wi,bi,30);    % short physics-consistent fine-tune
    T_inf(r)=toc; M_inf(r)=mean((Tc-netI(Xc)).^2);
end

%% Aggregate
methods={'Retraining','SISA','Fisher','Influence(Ours)'};
MSE=[mean(M_ret) mean(M_sisa) mean(M_fish) mean(M_inf)];
TIME=[mean(T_ret) mean(T_sisa) mean(T_fish) mean(T_inf)];
SPD=mean(T_ret)./TIME;
if ~exist('results','dir'); mkdir('results'); end
cmp=struct('methods',{methods},'MSE',MSE,'time',TIME,'speedup',SPD);
save(fullfile('results','comparison.mat'),'cmp');

fprintf('\n%-18s %-12s %-10s %-8s\n','Method','MSE','Time(s)','Speedup');
for i=1:numel(methods)
    fprintf('%-18s %.3e   %.4f    %.1fx\n',methods{i},MSE(i),TIME(i),SPD(i));
end

%% ---- helpers ----
function net=trainNet(X,T,arch,ep)
    net=feedforwardnet(arch); net.trainParam.epochs=ep; net.trainParam.goal=1e-8;
    net.trainParam.showWindow=false; net=configure(net,X,T); net=train(net,X,T);
end
function [W,b]=getWB(net,arch)
    nl=numel(arch)+1; W=cell(nl,1); b=cell(nl,1);
    W{1}=net.IW{1,1}; b{1}=net.b{1};
    for L=2:nl, W{L}=net.LW{L,L-1}; b{L}=net.b{L}; end
end
function g=accumGrad(W,b,X,T)
    nl=numel(W); g.W=cellfun(@(w) zeros(size(w)),W,'uni',0); g.b=cellfun(@(v) zeros(size(v)),b,'uni',0);
    for j=1:numel(T)
        [a,~]=fwd(W,b,X(j)); dz=2*(a{nl+1}-T(j));
        for L=nl:-1:1
            g.W{L}=g.W{L}+dz*a{L}'; g.b{L}=g.b{L}+dz;
            if L>1, dz=(W{L}'*dz).*(1-a{L}.^2); end
        end
    end
end
function Fd=accumFisher(W,b,X,T)
    nl=numel(W); Fd.W=cellfun(@(w) zeros(size(w)),W,'uni',0); Fd.b=cellfun(@(v) zeros(size(v)),b,'uni',0);
    for j=1:numel(T)
        [a,~]=fwd(W,b,X(j)); dz=2*(a{nl+1}-T(j));
        for L=nl:-1:1
            Fd.W{L}=Fd.W{L}+(dz*a{L}').^2; Fd.b{L}=Fd.b{L}+dz.^2;
            if L>1, dz=(W{L}'*dz).*(1-a{L}.^2); end
        end
    end
    Fd.W=cellfun(@(w) w/numel(T),Fd.W,'uni',0); Fd.b=cellfun(@(v) v/numel(T),Fd.b,'uni',0);
end
function [a,z]=fwd(W,b,x)
    nl=numel(W); a=cell(nl+1,1); z=cell(nl,1); a{1}=x;
    for L=1:nl
        z{L}=W{L}*a{L}+b{L};
        if L<nl, a{L+1}=tanh(z{L}); else, a{L+1}=z{L}; end
    end
end
function net=setWB(X,T,arch,W,b)
    net=feedforwardnet(arch); net=configure(net,X,T); nl=numel(arch)+1;
    net.IW{1,1}=W{1}; net.b{1}=b{1};
    for L=2:nl, net.LW{L,L-1}=W{L}; net.b{L}=b{L}; end
end
function net=setWBtrain(X,T,arch,W,b,ep)
    net=setWB(X,T,arch,W,b);
    net.trainParam.epochs=ep; net.trainParam.goal=1e-8; net.trainParam.showWindow=false;
    net=train(net,X,T);
end
