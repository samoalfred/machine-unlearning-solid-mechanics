%% Summarize the UQ ensembles (CS1-6): 95% prediction-interval coverage and
%% epistemic-std reduction (biased -> unlearned). Regenerates the beam
%% uncertainty figure (no in-figure title) and an appendix UQ bar figure.
clc; close all; DPI=200;
Uf=load(fullfile('results','uq_all_ff.mat')).U;
Up=load(fullfile('results','uq_all_pinn.mat')).U;
noise=containers.Map({1,2,3,4,5,6},{0.01,0.001,1e-4,0.02,0,0});
names={'Beam','Temperature','Multi-axial','Cyclic','Elasticity PINN','Neo-Hookean PINN'};

cases=[]; cov=[]; sb=[]; sa=[];
if isfield(Uf,'noise'), Uf=rmfield(Uf,'noise'); end
allU=[Uf Up];
for i=1:numel(allU)
    c=allU(i).case; Pb=allU(i).Pb; Pa=allU(i).Pa; Tt=allU(i).Tt; nz=noise(c);
    mA=mean(Pa,1); eA=std(Pa,0,1); half=1.96*sqrt(eA.^2+nz^2);
    coverage=mean(abs(Tt-mA)<=half);
    cases(end+1)=c; cov(end+1)=coverage; sb(end+1)=mean(std(Pb,0,1)); sa(end+1)=mean(std(Pa,0,1));
end
[cases,ord]=sort(cases); cov=cov(ord); sb=sb(ord); sa=sa(ord);
fprintf('\n=== UQ summary ===\n%-18s %10s %14s %14s %10s\n','Case','Coverage','EpiStd biased','EpiStd unlearn','Members');
mem=[10 10 10 10 5 5];
for i=1:numel(cases)
    fprintf('%-18s %9.1f%% %14.3e %14.3e %8d\n',names{cases(i)},100*cov(i),sb(i),sa(i),mem(i));
end

%% ---- beam uncertainty figure (case 1), no in-figure title ----
i1=find([allU.case]==1,1); Pb=allU(i1).Pb; Pa=allU(i1).Pa; Tt=allU(i1).Tt;
Fg=linspace(0,500,numel(Tt)); nz=0.01;
mB=mean(Pb,1); eB=std(Pb,0,1); mA=mean(Pa,1); eA=std(Pa,0,1);
figure('Position',[60 60 1100 440],'Color','w');
subplot(1,2,1); hold on; grid on;
fill([Fg fliplr(Fg)],[mB-1.96*sqrt(eB.^2+nz^2) fliplr(mB+1.96*sqrt(eB.^2+nz^2))],[1 0.8 0.8],'EdgeColor','none');
plot(Fg,Tt,'k-','LineWidth',2); plot(Fg,mB,'r--','LineWidth',1.5);
xlabel('Force F (N)'); ylabel('Deflection \delta'); title('(a) Biased ensemble'); legend('95% PI','True','Mean','Location','northwest');
subplot(1,2,2); hold on; grid on;
fill([Fg fliplr(Fg)],[mA-1.96*sqrt(eA.^2+nz^2) fliplr(mA+1.96*sqrt(eA.^2+nz^2))],[0.8 0.8 1],'EdgeColor','none');
plot(Fg,Tt,'k-','LineWidth',2); plot(Fg,mA,'m--','LineWidth',1.5);
xlabel('Force F (N)'); ylabel('Deflection \delta'); title('(b) Unlearned ensemble'); legend('95% PI','True','Mean','Location','northwest');
exportgraphics(gcf,'uncertainty.png','Resolution',DPI); fprintf('saved uncertainty.png\n');

%% ---- appendix UQ figure: coverage + epistemic-std reduction across cases ----
figure('Position',[60 60 1100 440],'Color','w');
subplot(1,2,1); bar(100*cov); grid on; set(gca,'XTickLabel',names(cases),'XTickLabelRotation',25);
hold on; yline(95,'k--','LineWidth',1.2); ylabel('95% PI coverage (%)'); title('(a) Unlearned-ensemble coverage'); ylim([0 105]);
subplot(1,2,2); bar([sb(:) sa(:)]); grid on; set(gca,'XTickLabel',names(cases),'XTickLabelRotation',25);
set(gca,'YScale','log'); ylabel('Mean epistemic std'); legend('Biased','Unlearned','Location','northeast'); title('(b) Epistemic uncertainty');
v=[sb(:);sa(:)]; ylim([10^floor(log10(min(v))) 10^ceil(log10(max(v)))]);   % show every bar on the log axis
exportgraphics(gcf,'uq_appendix.png','Resolution',DPI); fprintf('saved uq_appendix.png\n');
fprintf('Done.\n');
