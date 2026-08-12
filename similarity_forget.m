%% Valid-data forgetting vs material similarity (reviewer Question 2).
% Materials A (kA=100) and B (kB varied). Train on A+B, forget A, compare to a
% model retrained on B only. As kB -> kA the materials become more similar;
% we measure whether forgetting still works. No physics penalty (both valid).
clear; clc; close all;
rng(7);
Fmin=0; Fmax=500; nA=600; nB=600; arch=[20 15]; nSeed=5; alpha=1e-6; kA=100;
kBs=[50 70 90 95];
Fg=linspace(Fmin,Fmax,400);

d_full=zeros(numel(kBs),1); d_un=zeros(numel(kBs),1);
for j=1:numel(kBs)
    kB=kBs(j); df=zeros(nSeed,1); du=zeros(nSeed,1);
    for s=1:nSeed
        rng(400+10*j+s);
        FA=(Fmax-Fmin)*rand(nA,1)+Fmin; dA=FA/kA+0.01*randn(nA,1);
        FB=(Fmax-Fmin)*rand(nB,1)+Fmin; dB=FB/kB+0.01*randn(nB,1);
        XA=FA'; TA=dA'; XB=FB'; TB=dB'; Xall=[XA XB]; Tall=[TA TB];
        netFull=mk(Xall,Tall,arch,500); netB=mk(XB,TB,arch,500);
        [W,b]=getWB(netFull,arch); gA=accumGrad(W,b,XA,TA);
        netUn=setWBtrain(XB,TB,arch,subW(W,gA,alpha),b,30);
        fB=netB(Fg);
        df(s)=mean((netFull(Fg)-fB).^2); du(s)=mean((netUn(Fg)-fB).^2);
    end
    d_full(j)=mean(df); d_un(j)=mean(du);
    fprintf('kB=%3d (kB/kA=%.2f): dist A+B->B %.3e  unlearned->B %.3e  (%.0f%% closer)\n', ...
        kB, kB/kA, d_full(j), d_un(j), (1-d_un(j)/d_full(j))*100);
end
if ~exist('results','dir'); mkdir('results'); end
sf=struct('kA',kA,'kBs',kBs,'dist_full',d_full,'dist_unlearn',d_un); save(fullfile('results','similarity_forget.mat'),'sf');
fprintf('Saved results/similarity_forget.mat\n');

function net=mk(X,T,arch,ep); net=feedforwardnet(arch); net.trainParam.epochs=ep; net.trainParam.goal=1e-8;
    net.trainParam.showWindow=false; net=configure(net,X,T); net=train(net,X,T); end
function [W,b]=getWB(net,arch); nl=numel(arch)+1; W=cell(nl,1); b=cell(nl,1);
    W{1}=net.IW{1,1}; b{1}=net.b{1}; for L=2:nl, W{L}=net.LW{L,L-1}; b{L}=net.b{L}; end, end
function g=accumGrad(W,b,X,T); nl=numel(W); g=cellfun(@(w) zeros(size(w)),W,'uni',0);
    for j=1:numel(T), a=cell(nl+1,1); a{1}=X(j);
        for L=1:nl, z=W{L}*a{L}+b{L}; if L<nl,a{L+1}=tanh(z); else,a{L+1}=z; end, end
        dz=2*(a{nl+1}-T(j));
        for L=nl:-1:1, g{L}=g{L}+dz*a{L}'; if L>1, dz=(W{L}'*dz).*(1-a{L}.^2); end, end
    end
end
function o=subW(W,g,al); o=cellfun(@(w,gg) w-al*gg, W, g,'uni',0); end
function net=setWBtrain(X,T,arch,W,b,ep); net=feedforwardnet(arch); net=configure(net,X,T); nl=numel(arch)+1;
    net.IW{1,1}=W{1}; net.b{1}=b{1}; for L=2:nl, net.LW{L,L-1}=W{L}; net.b{L}=b{L}; end
    net.trainParam.epochs=ep; net.trainParam.goal=1e-8; net.trainParam.showWindow=false; net=train(net,X,T); end
