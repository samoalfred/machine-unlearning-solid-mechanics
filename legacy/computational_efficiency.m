%% Computational efficiency: aggregate MEASURED timings from all six problems
% Loads results/problemN.mat (written by each problem script) and plots the
% measured retraining vs unlearning time. No hard-coded timings.

clear; clc; close all;
files = {'problem1','problem2','problem3','problem4','problem5','problem6'};
labels = {'Beam','Temp-Dep','Multi-Axial','Cyclic','Elast. PINN','Hyperel. PINN'};

nP = numel(files);
retrain = zeros(nP,1); retrain_s = zeros(nP,1);
unlearn = zeros(nP,1); unlearn_s = zeros(nP,1);
speedup = zeros(nP,1); speedup_s = zeros(nP,1);

for i=1:nP
    S = load(fullfile('results',[files{i} '.mat'])); r = S.res;
    retrain(i)=r.time_retrain(1); retrain_s(i)=r.time_retrain(2);
    unlearn(i)=r.time_unlearn(1); unlearn_s(i)=r.time_unlearn(2);
    speedup(i)=r.speedup(1);      speedup_s(i)=r.speedup(2);
end

fprintf('%-14s %-16s %-16s %-10s\n','Problem','Retrain(s)','Unlearn(s)','Speedup');
for i=1:nP
    fprintf('%-14s %6.2f +/- %5.2f   %6.2f +/- %5.2f   %.1fx\n', ...
        labels{i}, retrain(i),retrain_s(i), unlearn(i),unlearn_s(i), speedup(i));
end

%% Figure: grouped bars (log scale) with speedup annotations
figure(1); clf; set(gcf,'Position',[50 50 950 550]);
bd = [retrain, unlearn];
bh = bar(bd); grid on; set(gca,'YScale','log');
bh(1).FaceColor=[0.2 0.4 0.8]; bh(2).FaceColor=[0.85 0.5 0.2];
set(gca,'XTickLabel',labels); ylabel('Wall-clock time (s, log scale)');
legend({'Full retraining','Unlearning'},'Location','northwest');
title('Measured computational cost: retraining vs unlearning');
ymax = max(retrain);
for i=1:nP
    text(i, retrain(i)*1.3, sprintf('%.1f\\times', speedup(i)), ...
        'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
end
ylim([min(unlearn)*0.3, ymax*3]);
saveas(gcf,'computational_efficiency.png');
fprintf('\nFigure saved to computational_efficiency.png\n');

eff = struct('labels',{labels},'retrain',retrain,'unlearn',unlearn,'speedup',speedup, ...
    'retrain_std',retrain_s,'unlearn_std',unlearn_s);
save(fullfile('results','efficiency.mat'),'eff');
