%% Re-plot the beam ablation figure from saved results (no in-figure title),
%% consistent (a)-(d) panels. MSE normalized to the no-corruption baseline.
clc; close all; DPI=200;
a=load(fullfile('results','ablation.mat')).ab; b=a.baseline;
figure('Position',[60 60 1100 800],'Color','w');
subplot(2,2,1); plot(a.n_forget,a.mse_forget(:)/b,'o-','LineWidth',1.6); grid on;
xlabel('Number of forgotten samples'); ylabel('MSE / baseline'); title('(a) Forgotten samples');
subplot(2,2,2); plot(a.corruption,a.mse_corr(:)/b,'s-','LineWidth',1.6); grid on;
xlabel('Corruption magnitude (\times true)'); ylabel('MSE / baseline'); title('(b) Corruption magnitude');
subplot(2,2,3); plot(a.depth,a.mse_depth(:)/b,'d-','LineWidth',1.6); grid on;
xlabel('Hidden layers'); ylabel('MSE / baseline'); title('(c) Network depth');
subplot(2,2,4); semilogx(a.alpha,a.mse_alpha(:)/b,'^-','LineWidth',1.6); grid on;
xlabel('Unlearning step size \alpha'); ylabel('MSE / baseline'); title('(d) Step size');
exportgraphics(gcf,'ablation.png','Resolution',DPI); fprintf('saved ablation.png\n');
