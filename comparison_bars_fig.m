%% Appendix figure: beam baseline comparison (Table in Appendix C).
%% (a) clean-test MSE, (b) speedup over full retraining, for retrain/SISA/
%% Fisher/proposed. Values from the 5-seed beam comparison.
%
% Hard-coded summary values (not recomputed here) plotted as two grouped bar
% charts and exported to the manuscript folder.
clear; clc; close all;

% ---- output path + data ----
OUT=['C:\Users\samoa\OneDrive - Umich\Documents\My Projects\Machine Unlearning\' ...
     'Revisions\Revised Manuscript\comparison_bars.png'];
methods={'Retrain','SISA','Fisher','Proposed'};
MSE=[9.4e-5 1.0e-4 1.2e-4 1.3e-4];   % clean-test MSE per method (5-seed beam comparison)
spd=[1.0 1.7 1.7 1.9];   % wall-clock speedup over full retraining
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

exportgraphics(gcf,OUT,'Resolution',300,'BackgroundColor','white');
fprintf('saved %s\n',OUT);
