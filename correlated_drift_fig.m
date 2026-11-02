%% Publication figure for the correlated sensor-drift study (manuscript 4.16).
%% (a) representative seed: true/biased/unlearned/retrained deflection with the
%% corrupted load band shaded; (b) global and band-local MSE (5-seed means).
%
% Panel (a) is recomputed here from one representative seed (train biased +
% retrained, then the proposed adaptive-Newton + fine-tune unlearning); panel (b)
% reads the 5-seed means from results/rev_correlated.mat.
clear; clc; close all;

% ---- paths + configuration ----
ROOT='C:\Users\samoa\OneDrive - Umich\Documents\My Projects\Machine Unlearning';
OUT=fullfile(ROOT,'Revisions','Revised Manuscript','correlated_drift.png');

layers=[1 20 15 1]; nL=numel(layers)-1; k=100; Fsc=500;   % network, beam stiffness, input scale
adamIters=1500; Tft=80; cgIters=60; gammaFrac=1e-3;   % train/fine-tune/CG iters, damping fraction
Fd=400; Wd=60; Adrift=0.45*(Fd/k);   % drift band center, half-width, and drift amplitude

% ---- representative seed: build the corrupted data ----
rng(901);
F=Fsc*rand(1,1000); d=F/k+0.01*randn(1,1000);   % clean beam data
inband=abs(F-Fd)<=Wd;   % points inside the corrupted load band
drift=Adrift*exp(-((F-Fd)/(0.6*Wd)).^2);   % smooth drift bump
lfn=0.15*(Fd/k)*sin(2*pi*(F-Fd)/(2*Wd));   % correlated low-frequency wobble
dcorr=d; dcorr(inband)=d(inband)+drift(inband)+lfn(inband);   % corrupt only the in-band targets
X=F/Fsc; Tcorr=dcorr; Xb=X(inband); Tb=Tcorr(inband);   % full inputs; corrupted subset
keep=~inband; Xk=X(keep); Tk=d(keep);   % retained (clean) points
Fg=linspace(0,Fsc,600); Xt=Fg/Fsc; ytrue=Fg/k;   % dense test grid + true curve

% ---- train, unlearn, predict ----
pB=trainAdam(layers,dlarray(X),dlarray(Tcorr),adamIters);   % biased
pR=trainAdam(layers,dlarray(Xk),dlarray(Tk),adamIters);   % retrained (clean only)
Ntot=numel(X);
g=dlfeval(@gradSubset,pB,dlarray(Xb),dlarray(Tb),Ntot,nL);   % g_bad
pN=adaptiveNewton(pB,dlarray(X),dlarray(Tcorr),dlarray(Xk),dlarray(Tk),g,gammaFrac,cgIters,nL);   % Newton influence step
pU=fineTune(pN,dlarray(Xk),dlarray(Tk),Tft,nL);   % + clean fine-tune -> unlearned
yb=pred(pB,Xt,nL); yu=pred(pU,Xt,nL); yr=pred(pR,Xt,nL);   % predictions for panel (a)

% ---- 5-seed means for panel (b) ----
L=load(fullfile(ROOT,'results','rev_correlated.mat')); CR=L.CR;
G=[mean(CR.mse_b) mean(CR.mse_u) mean(CR.mse_r)];   % global test MSE: biased/unlearned/retrained
B=[mean(CR.band_b) mean(CR.band_u) mean(CR.band_r)];   % band-local MSE: biased/unlearned/retrained

figure('Units','inches','Position',[1 1 10 4],'Color','w');

% ---- panel (a): curves + shaded corrupted band ----
subplot(1,2,1); hold on; box on;
patch([Fd-Wd Fd+Wd Fd+Wd Fd-Wd],[0 0 7 7],[1 0.92 0.65],'EdgeColor','none');   % shaded removed band
plot(F(inband),dcorr(inband),'.','Color',[0.75 0.2 0.2],'MarkerSize',6);   % corrupted points
h1=plot(Fg,ytrue,'k-','LineWidth',2);
h2=plot(Fg,yb,'--','Color',[0.85 0.20 0.10],'LineWidth',1.6);
h3=plot(Fg,yu,'-.','Color',[0.00 0.45 0.74],'LineWidth',1.6);
h4=plot(Fg,yr,':','Color',[0.10 0.60 0.20],'LineWidth',1.6);
xlabel('Force F (N)'); ylabel('Deflection \delta'); ylim([0 7]); xlim([0 500]);
title('(a) Correlated sensor-drift corruption and recovery');
legend([h1 h2 h3 h4],{'True','Biased','Unlearned','Retrained'},'Location','northwest');
text(Fd,6.6,'removed band','HorizontalAlignment','center','FontSize',9,'Color',[0.6 0.45 0.1]);

% ---- panel (b): grouped log-scale MSE bars ----
subplot(1,2,2);
Y=[G; B];
hb=bar(Y,'grouped'); set(gca,'YScale','log');
cols=[0.85 0.20 0.10; 0.00 0.45 0.74; 0.10 0.60 0.20];   % biased/unlearned/retrained colors
for i=1:3, hb(i).FaceColor=cols(i,:); end
set(gca,'XTickLabel',{'Global','Band-local'}); ylabel('Test MSE (5-seed mean)');
ylim([1e-4 2]); grid on;
title('(b) Clean-test error: biased, unlearned, retrained');
legend({'Biased','Unlearned','Retrained'},'Location','northeast');

exportgraphics(gcf,OUT,'Resolution',300,'BackgroundColor','white');
fprintf('saved %s\n',OUT);
fprintf('global b/u/r = %.2e/%.2e/%.2e ; band b/u/r = %.2e/%.2e/%.2e\n',G,B);

%% ---- helpers (from rev_correlated.m) ----
% mlp: forward pass -- tanh hidden layers, linear output
function Y=mlp(X,p,L); A=X; for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L)); end
% initP: He-style weight init, zero biases, as dlarrays
function p=initP(layers); p=struct;
    for i=1:numel(layers)-1, p.(sprintf('W%d',i))=dlarray(sqrt(2/layers(i))*randn(layers(i+1),layers(i)));
        p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1)); end
end
% trainAdam: Adam loop with a stepped learning rate (5e-3 -> 1e-3 -> 2e-4)
function p=trainAdam(layers,X,T,iters); p=initP(layers); aG=[];aS=[];lr=5e-3;nL=numel(layers)-1;
    for it=1:iters, [~,g]=dlfeval(@lg,p,X,T,nL); if it==round(0.6*iters),lr=1e-3;end; if it==round(0.85*iters),lr=2e-4;end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
% fineTune: like trainAdam but starts from given params (iters==0 is a no-op)
function p=fineTune(p,X,T,iters,nL); if iters==0, return; end
    aG=[];aS=[];lr=1e-3;
    for it=1:iters, [~,g]=dlfeval(@lg,p,X,T,nL); if it==round(0.7*iters),lr=2e-4;end; [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
% lg: mean-squared loss and its gradient
function [l,g]=lg(p,X,T,nL); l=mean((mlp(X,p,nL)-T).^2,'all'); g=dlgradient(l,p); end
% gradSubset: g_bad = gradient of (sum of subset squared errors)/Ntot
function g=gradSubset(p,Xs,Ts,Ntot,nL); l=sum((mlp(Xs,p,nL)-Ts).^2,'all')/Ntot; g=dlgradient(l,p); end
% hvp: Hessian-vector product via double backprop (no H formed)
function Hv=hvp(p,v,X,T,nL)
    g=dlgradient(mean((mlp(X,p,nL)-T).^2,'all'),p,'EnableHigherDerivatives',true);
    gv=0; f=fieldnames(g); for i=1:numel(f), gv=gv+sum(g.(f{i}).*v.(f{i}),'all'); end; Hv=dlgradient(gv,p);
end
% powerLambda: power iteration for the top Hessian eigenvalue
function lam=powerLambda(p,X,T,nL,iters); v=randLike(p); v=nrm(v);
    for it=1:iters, Hv=dlfeval(@hvp,p,v,X,T,nL); lam=sdot(v,Hv); v=nrm(Hv); end
end
% cgSolve: damped conjugate gradient for (H + lam I) v = g, with a curvature guard
function v=cgSolve(p,g,X,T,lam,maxit,tol,nL)
    x=scal(g,0); r=g; pp=r; rs=sdot(r,r); g0=sqrt(rs);
    for it=1:maxit, Hp=dlfeval(@hvp,p,pp,X,T,nL); Ap=axpy(Hp,lam,pp);
        den=sdot(pp,Ap); if den<=0, den=den+lam*sdot(pp,pp)+1e-12; end   % guard against non-positive curvature
        al=rs/den; x=axpy(x,al,pp); r=axpy(r,-al,Ap); rs2=sdot(r,r); if sqrt(rs2)<tol*g0, break; end
        pp=axpy(r,rs2/rs,pp); rs=rs2; end
    v=x;
end
% adaptiveNewton: Newton step with damping increased until the retained fit stays safe
function pN=adaptiveNewton(pB,X,T,Xc,Tc,g,gf0,cgIters,nL)
    lam=powerLambda(pB,X,T,nL,25);   % base damping scale
    Lb=double(gather(extractdata(mean((mlp(Xc,pB,nL)-Tc).^2,'all'))));   % baseline retained-data loss
    gf=gf0; cand=pB;
    for tryi=1:7
        v=cgSolve(pB,g,X,T,gf*lam,cgIters,1e-6,nL); cand=axpy(pB,1.0,v);
        Lc=double(gather(extractdata(mean((mlp(Xc,cand,nL)-Tc).^2,'all'))));
        if isfinite(Lc) && Lc<5*Lb, pN=cand; return; end   % accept if the step didn't blow up the fit
        gf=gf*10;   % otherwise increase damping and retry
    end
    pN=cand;
end
% pred + struct-vector algebra
function y=pred(p,X,nL); y=extractdata(mlp(dlarray(X),p,nL)); end
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end   % a + al*b
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end   % c*a
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end   % inner product
function v=randLike(p); v=p; f=fieldnames(p); for i=1:numel(f), v.(f{i})=dlarray(randn(size(p.(f{i})))); end, end   % random vector shaped like p
function v=nrm(v); n=sqrt(sdot(v,v)); f=fieldnames(v); for i=1:numel(f), v.(f{i})=v.(f{i})/n; end, end   % normalize to unit length
