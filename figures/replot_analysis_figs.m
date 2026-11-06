%% Re-plot the efficiency and scaling figures from saved results, with NO
%% in-figure titles (captions live in the manuscript). No recomputation, so
%% the figures stay consistent with the reported tables.
clc; close all; DPI=200;
E=load(fullfile('results','efficiency.mat')).eff;
S=load(fullfile('results','scaling.mat')).sc;

% ---- computational_efficiency.png : grouped bars retrain vs unlearn ----
figure('Position',[60 60 900 460],'Color','w');
Y=[E.retrain(:) E.unlearn(:)]; b=bar(Y); hold on; grid on;
ng=size(Y,1); x1=(1:ng)-0.145; x2=(1:ng)+0.145;
errorbar(x1,E.retrain(:),E.retrain_std(:),'k.','LineWidth',1.2);
errorbar(x2,E.unlearn(:),E.unlearn_std(:),'k.','LineWidth',1.2);
ytop=max(Y,[],2);
for i=1:ng, text(i,ytop(i)*1.06,sprintf('%.1f\\times',E.speedup(i)),'HorizontalAlignment','center','FontSize',9); end
set(gca,'XTickLabel',E.labels,'XTickLabelRotation',20); ylabel('Wall-clock time (s)');
legend('Retraining','Unlearning','Location','northwest'); ylim([0 max(ytop)*1.2]);
exportgraphics(gcf,'computational_efficiency.png','Resolution',DPI); fprintf('saved computational_efficiency.png\n');

% ---- scaling.png : (a) cost vs params, (b) speedup vs params ----
figure('Position',[60 60 1100 440],'Color','w');
subplot(1,2,1); loglog(S.params,S.t_cold,'o-','LineWidth',1.6); hold on; grid on;
loglog(S.params,S.t_warm,'s-','LineWidth',1.6);
xlabel('Number of parameters'); ylabel('Wall-clock time (s)');
title('(a) Cost versus model size'); legend('Cold retrain','Warm-start unlearn','Location','northwest');
subplot(1,2,2); semilogx(S.params,S.speedup,'d-','LineWidth',1.6,'Color',[0.85 0.33 0.1]); grid on;
xlabel('Number of parameters'); ylabel('Warm-start speedup ($\times$)','Interpreter','latex');
title('(b) Speedup versus model size');
exportgraphics(gcf,'scaling.png','Resolution',DPI); fprintf('saved scaling.png\n');
fprintf('Done.\n');
