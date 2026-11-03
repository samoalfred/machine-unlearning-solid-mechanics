%% Bar-chart version of Table 10 (beam baseline comparison); not in the final manuscript.
%% (a) clean-test MSE and (b) wall-clock speedup over full retraining, for full
%% retraining, SISA, Fisher forgetting, and the first-order step + fine-tune.
%% Reads results/comparison.mat (written by comparison_script.m) and saves
%% comparison_bars.png in the current folder.
clear; clc; close all;

f=fullfile('results','comparison.mat');
if ~exist(f,'file'), error('Run comparison_script.m first (it writes %s).',f); end
cmp=load(f).cmp;
MSE=cmp.MSE; spd=cmp.speedup;   % order: retraining, SISA, Fisher, first-order + fine-tune
methods={'Retrain','SISA','Fisher','First-order + FT'};
cols=[0.45 0.45 0.45; 0.30 0.55 0.80; 0.90 0.60 0.20; 0.00 0.45 0.74];   % per-method bar colors

figure('Units','inches','Position',[1 1 10 4],'Color','w');

% ---- (a) accuracy ----
subplot(1,2,1); hold on; box on;
for i=1:4, bar(i,MSE(i),0.6,'FaceColor',cols(i,:)); end
set(gca,'XTick',1:4,'XTickLabel',methods); ylabel('Clean-test MSE');
title('(a) Accuracy on the beam'); grid on; ylim([0 1.5e-4]);
for i=1:4, text(i,MSE(i)+6e-6,sprintf('%.1f\\times10^{-5}',MSE(i)*1e5), ...
        'HorizontalAlignment','center','FontSize',8); end   % value labels above bars

% ---- (b) speed ----
subplot(1,2,2); hold on; box on;
for i=1:4, bar(i,spd(i),0.6,'FaceColor',cols(i,:)); end
set(gca,'XTick',1:4,'XTickLabel',methods); ylabel('Speedup over full retraining');
title('(b) Speed on the beam'); grid on; ylim([0 2.3]);
for i=1:4, text(i,spd(i)+0.07,sprintf('%.1f\\times',spd(i)), ...
        'HorizontalAlignment','center','FontSize',8); end   % value labels above bars

exportgraphics(gcf,'comparison_bars.png','Resolution',300,'BackgroundColor','white');
fprintf('saved comparison_bars.png\n');
