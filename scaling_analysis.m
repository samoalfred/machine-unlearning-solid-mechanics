%% Scaling measurement: cold retrain vs warm-start unlearning across model sizes
% Honest, small sweep to characterize (not overclaim) how the warm-start
% advantage behaves as the network grows. Trained with scaled conjugate
% gradient (scales better than Levenberg-Marquardt for larger nets).
clear; clc; close all;
rng(3);
k=100; Fmin=0; Fmax=500; Nclean=2000; Nbad=100; nRun=3;
archs={10,[50 50],[150 150],[300 300]};

nP=numel(archs);
pcount=zeros(nP,1);
t_cold=zeros(nP,1); t_warm=zeros(nP,1);
e_cold=zeros(nP,1); e_warm=zeros(nP,1);

for a=1:nP
    arch=archs{a};
    tc=zeros(nRun,1); tw=zeros(nRun,1); ec=zeros(nRun,1); ew=zeros(nRun,1);
    for r=1:nRun
        rng(50+a*10+r);
        F=(Fmax-Fmin)*rand(Nclean,1)+Fmin; d=F/k+0.01*randn(Nclean,1);
        Xc=F'; Tc=d';
        Fb=(Fmax-Fmin)*rand(Nbad,1)+Fmin; db=2*(Fb/k);
        Xf=[Xc Fb']; Tf=[Tc db'];

        % biased model (starting point for warm unlearn)
        netB=mk(Xf,Tf,arch,300,false);

        % cold retrain from scratch on clean data (gold standard)
        tic; [netR,tr]=mk(Xc,Tc,arch,300,true); tc(r)=toc; ec(r)=numel(tr.epoch);

        % warm-start unlearn: fine-tune biased on clean data (short budget)
        tic; [~,tw2]=fineTune(netB,Xc,Tc,50); tw(r)=toc; ew(r)=numel(tw2.epoch);

        if r==1, pcount(a)=countParams(netB); end
    end
    t_cold(a)=mean(tc); t_warm(a)=mean(tw); e_cold(a)=mean(ec); e_warm(a)=mean(ew);
end

fprintf('\n=== Scaling: cold retrain vs warm-start unlearn ===\n');
fprintf('%-10s %-10s %-14s %-14s %-8s\n','Arch','Params','ColdRetrain(s)','WarmUnlearn(s)','Speedup');
for a=1:nP
    fprintf('%-10s %-10d %-14.3f %-14.3f %-8.1fx\n', mat2str(archs{a}), pcount(a), ...
        t_cold(a), t_warm(a), t_cold(a)/t_warm(a));
end
fprintf('\nEpochs to converge (cold) vs fixed warm budget:\n');
for a=1:nP
    fprintf('  params %-8d cold %.0f epochs, warm %.0f epochs\n', pcount(a), e_cold(a), e_warm(a));
end

if ~exist('results','dir'); mkdir('results'); end
sc=struct('params',pcount,'t_cold',t_cold,'t_warm',t_warm,'e_cold',e_cold,'e_warm',e_warm, ...
    'speedup',t_cold./t_warm);
save(fullfile('results','scaling.mat'),'sc');
fprintf('\nSaved results/scaling.mat\n');

function [net,tr]=mk(X,T,arch,ep,doTrain)
    net=feedforwardnet(arch,'trainscg'); net.trainParam.epochs=ep; net.trainParam.goal=1e-7;
    net.trainParam.showWindow=false; net=configure(net,X,T);
    if nargin<5 || doTrain, [net,tr]=train(net,X,T); else, [net,tr]=train(net,X,T); end
end
function [net,tr]=fineTune(net,X,T,ep)
    net.trainParam.epochs=ep; net.trainParam.goal=1e-7; net.trainParam.showWindow=false;
    [net,tr]=train(net,X,T);
end
function n=countParams(net)
    n=numel(net.IW{1,1})+numel(net.b{1});
    for L=2:numel(net.b), n=n+numel(net.LW{L,L-1})+numel(net.b{L}); end
end
