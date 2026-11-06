%% Re-plot the beam ablation figure (Figure 13) from saved results.
%% Post-unlearning retained-data MSE normalized to the no-corruption baseline;
%% the dashed line at 1 marks "as accurate as a model that never saw the bad data".
%% Reads results/ablation.mat (written by ablation_script.m), saves ablation.png.
clc; close all;
if ~exist('outFile','var'), outFile='ablation.png'; end
a=load(fullfile('results','ablation.mat')).ab; b=a.baseline;
f=figure('Units','inches','Position',[1 1 10 7.6],'Color','w');
tl=tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');
c=[0 0.45 0.74]; lw=1.8; ms=7; fs=12;
yl='MSE / baseline';   % retained-data MSE (see caption)

ax=nexttile(tl); plot(ax,a.n_forget,a.mse_forget(:)/b,'o-','Color',c,'LineWidth',lw,'MarkerSize',ms); hold(ax,'on');
yline(ax,1,'k--'); grid(ax,'on'); set(ax,'FontSize',fs);
xlabel(ax,'Number of removed samples'); ylabel(ax,yl); title(ax,'(a) Removed samples','FontWeight','normal');

ax=nexttile(tl); plot(ax,a.corruption,a.mse_corr(:)/b,'s-','Color',c,'LineWidth',lw,'MarkerSize',ms); hold(ax,'on');
yline(ax,1,'k--'); grid(ax,'on'); set(ax,'FontSize',fs);
xlabel(ax,'Corruption magnitude (\times true value)'); ylabel(ax,yl); title(ax,'(b) Corruption magnitude','FontWeight','normal');

ax=nexttile(tl); plot(ax,a.depth,a.mse_depth(:)/b,'d-','Color',c,'LineWidth',lw,'MarkerSize',ms); hold(ax,'on');
yline(ax,1,'k--'); grid(ax,'on'); set(ax,'FontSize',fs,'XTick',a.depth);
xlabel(ax,'Number of hidden layers'); ylabel(ax,yl); title(ax,'(c) Network depth','FontWeight','normal');

ax=nexttile(tl); semilogx(ax,a.alpha,a.mse_alpha(:)/b,'^-','Color',c,'LineWidth',lw,'MarkerSize',ms); hold(ax,'on');
yline(ax,1,'k--'); grid(ax,'on'); set(ax,'FontSize',fs);
xlabel(ax,'First-order step size \alpha'); ylabel(ax,yl); title(ax,'(d) First-order step size','FontWeight','normal');

for ax=tl.Children', yl0=ylim(ax); ylim(ax,[min(0.9,yl0(1)) yl0(2)]); end
exportgraphics(f,outFile,'Resolution',300,'BackgroundColor','white');
fprintf('saved %s\n',outFile);
