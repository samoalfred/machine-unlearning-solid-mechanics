%% Problem 6: Neo-Hookean Hyperelasticity PINN with Corrupted Boundary Data
% Genuine physics-informed neural network (automatic differentiation).
% Network maps (x,y) -> (ux,uy). Finite-deformation equilibrium Div(P)=0 is
% enforced at collocation points, where the first Piola-Kirchhoff stress for a
% compressible Neo-Hookean solid is
%   F = I + Grad u,  J = det F,
%   P = mu (F - F^{-T}) + lambda ln(J) F^{-T}.
% All MSE values and residual maps are COMPUTED by AD (no hard-coded numbers).

clear; clc; close all;
rng(42);

%% Setup (nondimensional; small far-field load keeps J>0)
L = 2; R = 0.5;
mu = 1.0; lambda = 1.0;    % nondimensional Lame-type parameters
load_amp = 0.05;           % far-field load amplitude
N_coll = 300; N_bnd = 200; N_bad = 40;
layers = [2 30 30 30 2];
numRuns = 5; iters = 220; ftiters = 80; maxSeedsPerCall = 2;

uref = @(x,y) deal( load_amp * x .* (1 + 0.5*R^2 ./ (x.^2+y.^2)), ...
                   -load_amp * y .* (1 + 0.5*R^2 ./ (x.^2+y.^2)) );

fprintf('=== Problem 6: Neo-Hookean Hyperelasticity PINN (AD) ===\n');

if ~exist('results','dir'); mkdir('results'); end
partialFile = fullfile('results','problem6_partial.mat');
if exist(partialFile,'file')
    A=load(partialFile);
    mse_c=A.mse_c; mse_b=A.mse_b; mse_u=A.mse_u;
    t_re=A.t_re; t_un=A.t_un; res_c=A.res_c; res_b=A.res_b; res_u=A.res_u;
else
    mse_c=[]; mse_b=[]; mse_u=[]; t_re=[]; t_un=[]; res_c=[]; res_b=[]; res_u=[];
end
done=numel(mse_c);
fprintf('Seeds already completed: %d / %d\n', done, numRuns);

ranThisCall=0;
for run=(done+1):numRuns
    if ranThisCall>=maxSeedsPerCall, break; end
    ranThisCall=ranThisCall+1;
    rng(60+run);
    Xc = sampleDomain(N_coll,L,R);
    th = 2*pi*rand(1,N_bnd/2); Xh=[R*cos(th);R*sin(th)];
    Xo = outerEdges(N_bnd-N_bnd/2,L); Xb=[Xh,Xo];
    [ubx,uby]=uref(Xb(1,:),Xb(2,:)); Ub=[ubx;uby];
    badIdx = randperm(size(Xh,2),N_bad);
    Ub_bad = Ub;
    Ub_bad(1,badIdx)=Ub_bad(1,badIdx)+0.15;   % corrupted traction/displacement
    Ub_bad(2,badIdx)=Ub_bad(2,badIdx)+0.10;
    Xt = sampleDomain(1500,L,R); [utx,uty]=uref(Xt(1,:),Xt(2,:)); Ut=[utx;uty];

    tic;
    pC = trainPINN(layers,Xc,Xb,Ub,mu,lambda,L,iters); t_re(run)=toc;
    mse_c(run)=predMSE(pC,Xt,Ut,layers,L); res_c(run)=meanResidual(pC,Xc,mu,lambda,L,layers);

    pB = trainPINN(layers,Xc,Xb,Ub_bad,mu,lambda,L,iters);
    mse_b(run)=predMSE(pB,Xt,Ut,layers,L); res_b(run)=meanResidual(pB,Xc,mu,lambda,L,layers);

    tic;
    pU = unlearnPINN(pB,Xb(:,badIdx),Ub_bad(:,badIdx),Xc,Xb,Ub,badIdx,mu,lambda,L,layers,ftiters);
    t_un(run)=toc;
    mse_u(run)=predMSE(pU,Xt,Ut,layers,L); res_u(run)=meanResidual(pU,Xc,mu,lambda,L,layers);

    fprintf('run %d: MSE clean %.3e biased %.3e unlearned %.3e | res b/u %.3e/%.3e\n', ...
        run,mse_c(run),mse_b(run),mse_u(run),res_b(run),res_u(run));
    save(partialFile,'mse_c','mse_b','mse_u','t_re','t_un','res_c','res_b','res_u');
end

if numel(mse_c) < numRuns
    fprintf('Completed %d/%d seeds this session; re-run to accumulate more.\n', numel(mse_c), numRuns);
    return;
end

errRed=(mean(mse_b)-mean(mse_u))/mean(mse_b)*100;
[~,pval]=ttest(mse_b,mse_u); dcoh=(mean(mse_b)-mean(mse_u))/std(mse_b-mse_u);
fprintf('\nClean %.3e Biased %.3e Unlearned %.3e (ErrRed %.1f%%)\n',mean(mse_c),mean(mse_b),mean(mse_u),errRed);
fprintf('Mean residual clean/biased/unlearned: %.3e / %.3e / %.3e\n',mean(res_c),mean(res_b),mean(res_u));

if ~exist('results','dir'); mkdir('results'); end
res=struct('name','Hyperelasticity PINN', ...
    'mse_clean',[mean(mse_c) std(mse_c)],'mse_biased',[mean(mse_b) std(mse_b)], ...
    'mse_unlearned',[mean(mse_u) std(mse_u)],'err_reduction',errRed, ...
    'time_retrain',[mean(t_re) std(t_re)],'time_unlearn',[mean(t_un) std(t_un)], ...
    'speedup',[mean(t_re./t_un) std(t_re./t_un)], ...
    'residual',[mean(res_c) mean(res_b) mean(res_u)],'p_value',pval,'cohen_d',dcoh, ...
    'nRuns',numRuns,'mse_biased_all',mse_b,'mse_unlearned_all',mse_u);
save(fullfile('results','problem6.mat'),'res');
fprintf('Saved results/problem6.mat\n');

%% Figure 1: points + boundary + performance
figure(1); clf; set(gcf,'Position',[50 50 1200 800]);
th2=linspace(0,2*pi,100);
subplot(2,2,1);
plot(Xc(1,:),Xc(2,:),'b.','MarkerSize',6); hold on;
plot(Xb(1,:),Xb(2,:),'ko','MarkerSize',5);
plot(Xh(1,badIdx),Xh(2,badIdx),'rx','MarkerSize',10,'LineWidth',2);
plot(R*cos(th2),R*sin(th2),'k-','LineWidth',1.5);
axis equal; grid on; xlim([-L L]); ylim([-L L]);
xlabel('x'); ylabel('y'); title('(a) Points'); legend('Collocation','Boundary','Corrupted','Location','best');
subplot(2,2,2);
scatter(Xb(1,:),Ub_bad(1,:),40,'ko','filled'); hold on;
scatter(Xh(1,badIdx),Ub_bad(1,badIdx),60,'rx','LineWidth',2);
xlabel('x'); ylabel('u_x'); title('(b) u_x'); grid on;
subplot(2,2,3);
scatter(Xb(2,:),Ub_bad(2,:),40,'ko','filled'); hold on;
scatter(Xh(2,badIdx),Ub_bad(2,badIdx),60,'rx','LineWidth',2);
xlabel('y'); ylabel('u_y'); title('(c) u_y'); grid on;
subplot(2,2,4);
errs=[mean(mse_c) mean(mse_b) mean(mse_u)]; es=[std(mse_c) std(mse_b) std(mse_u)];
bar(errs); hold on; errorbar(1:3,errs,es,'k.','LineWidth',2);
set(gca,'XTickLabel',{'Clean','Biased','Unlearned'}); ylabel('MSE'); title('(d) Performance'); grid on;
sgtitle('Problem 6: Neo-Hookean Hyperelasticity PINN','FontSize',14);
saveas(gcf,'problem6_results.png'); fprintf('Figure saved to problem6_results.png\n');

%% Figure 2: displacement magnitude fields
figure(2); clf; set(gcf,'Position',[50 50 1200 800]);
gx=linspace(-L,L,40); [Gx,Gy]=meshgrid(gx,gx); m=sqrt(Gx.^2+Gy.^2)>R; Xg=[Gx(m)';Gy(m)'];
UC=predField(pC,Xg,layers,L); UB=predField(pB,Xg,layers,L); UU=predField(pU,Xg,layers,L);
magf=@(U) hypot(U(1,:),U(2,:));
plotField=@(vals,ttl) deal(vals,ttl);
titles={'(a) Clean','(b) Biased','(c) Unlearned','(d) |Biased-Clean| error'};
mags={magf(UC),magf(UB),magf(UU),abs(magf(UB)-magf(UC))};
for k=1:4
    subplot(2,2,k); Z=nan(size(Gx)); Z(m)=mags{k};
    imagesc(gx,gx,Z); set(gca,'YDir','normal'); hold on;
    plot(R*cos(th2),R*sin(th2),'w-','LineWidth',2);
    axis equal; axis tight; colorbar; title(titles{k}); xlabel('x'); ylabel('y');
end
sgtitle('Displacement magnitude (Neo-Hookean PINN)','FontSize',14);
saveas(gcf,'problem6_displacement.png'); fprintf('Figure saved to problem6_displacement.png\n');

%% Figure 3: REAL residual maps
figure(3); clf; set(gcf,'Position',[50 50 1200 400]);
Rc=residualField(pC,Xg,mu,lambda,L,layers);
Rb=residualField(pB,Xg,mu,lambda,L,layers);
Ru=residualField(pU,Xg,mu,lambda,L,layers);
cl=[0 max(Rb)]; Rs={Rc,Rb,Ru}; tt={'(a) Clean','(b) Biased','(c) Unlearned'};
for k=1:3
    subplot(1,3,k); Z=nan(size(Gx)); Z(m)=Rs{k};
    imagesc(gx,gx,Z); set(gca,'YDir','normal'); hold on;
    plot(R*cos(th2),R*sin(th2),'w-','LineWidth',2);
    axis equal; axis tight; caxis(cl); colorbar; title(tt{k}); xlabel('x'); ylabel('y');
end
sgtitle('PDE equilibrium residual |Div P| (computed by AD)','FontSize',14);
saveas(gcf,'problem6_residuals.png'); fprintf('Figure saved to problem6_residuals.png\n');

%% ================= Helpers =================
function X=sampleDomain(N,L,R)
    X=zeros(2,N); c=0;
    while c<N, p=-L+2*L*rand(2,1); if norm(p)>R, c=c+1; X(:,c)=p; end, end
end
function X=outerEdges(N,L)
    X=zeros(2,N);
    for i=1:N
        s=randi(4); t=-L+2*L*rand();
        switch s, case 1, X(:,i)=[t;L]; case 2, X(:,i)=[t;-L]; case 3, X(:,i)=[L;t]; case 4, X(:,i)=[-L;t]; end
    end
end
function Y=mlp(X,p,L)
    A=X; for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L));
end
function p=initParams(layers)
    % Small output-layer scale keeps initial displacements ~0 so that
    % F = I + grad u has det(F) > 0 (avoids complex log(J) in Neo-Hookean).
    p=struct; nL=numel(layers)-1;
    for i=1:nL
        s = sqrt(2/layers(i)); if i==nL, s = 0.01; end
        p.(sprintf('W%d',i))=dlarray(s*randn(layers(i+1),layers(i)));
        p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1));
    end
end
function p=trainPINN(layers,Xc,Xb,Ub,mu,lam,L,iters)
    p=initParams(layers); Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub);
    avgG=[]; avgSq=[]; lr=2e-3;
    for it=1:iters
        [~,g]=dlfeval(@pinnLoss,p,Xcd,Xbd,Ubd,mu,lam,L,numel(layers)-1);
        if it==round(0.7*iters), lr=5e-4; end
        [p,avgG,avgSq]=adamupdate(p,g,avgG,avgSq,it,lr);
    end
end
function [loss,grad]=pinnLoss(p,Xc,Xb,Ub,mu,lam,L,nL)
    dataLoss=mean((mlp(Xb,p,nL)-Ub).^2,'all');
    [rx,ry]=equilibrium(Xc,p,mu,lam,L,nL);
    physLoss=mean(rx.^2+ry.^2,'all');
    loss=dataLoss+3e-5*physLoss; grad=dlgradient(loss,p);
end
function [rx,ry]=equilibrium(Xs,p,mu,lam,L,nL)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L;
    duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    a=1+dux(1,:); b=dux(2,:); c=duy(1,:); d=1+duy(2,:);   % F = I + grad u
    J=a.*d-b.*c;
    % F^{-T} components
    iTxx= d./J; iTxy=-c./J; iTyx=-b./J; iTyy= a./J;
    logJ=log(J);
    Pxx=mu*(a-iTxx)+lam*logJ.*iTxx;
    Pxy=mu*(b-iTxy)+lam*logJ.*iTxy;
    Pyx=mu*(c-iTyx)+lam*logJ.*iTyx;
    Pyy=mu*(d-iTyy)+lam*logJ.*iTyy;
    dPxx=dlgradient(sum(Pxx,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dPxy=dlgradient(sum(Pxy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dPyx=dlgradient(sum(Pyx,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dPyy=dlgradient(sum(Pyy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    rx=dPxx(1,:)+dPxy(2,:);
    ry=dPyx(1,:)+dPyy(2,:);
end
function pU=unlearnPINN(pB,Xbad,Ubad,Xc,Xb,Ub,badIdx,mu,lam,L,layers,ftiters)
    nL=numel(layers)-1; Xbd=dlarray(Xbad/L); Ubd=dlarray(Ubad);
    [~,g]=dlfeval(@(pp) dataOnly(pp,Xbd,Ubd,nL),pB);
    pU=pB; al=5e-4;
    for i=1:nL
        pU.(sprintf('W%d',i))=pU.(sprintf('W%d',i))+al*g.(sprintf('W%d',i));
        pU.(sprintf('b%d',i))=pU.(sprintf('b%d',i))+al*g.(sprintf('b%d',i));
    end
    keep=true(1,size(Xb,2)); keep(badIdx)=false;
    Xbk=dlarray(Xb(:,keep)/L); Ubk=dlarray(Ub(:,keep)); Xcd=dlarray(Xc/L);
    avgG=[]; avgSq=[]; lr=5e-4;
    for it=1:ftiters
        [~,gg]=dlfeval(@pinnLoss,pU,Xcd,Xbk,Ubk,mu,lam,L,nL);
        [pU,avgG,avgSq]=adamupdate(pU,gg,avgG,avgSq,it,lr);
    end
end
function [loss,grad]=dataOnly(p,Xb,Ub,nL)
    loss=mean((mlp(Xb,p,nL)-Ub).^2,'all'); grad=dlgradient(loss,p);
end
function mse=predMSE(p,Xt,Ut,layers,L)
    U=extractdata(mlp(dlarray(Xt/L),p,numel(layers)-1)); mse=mean((U(:)-Ut(:)).^2);
end
function U=predField(p,Xg,layers,L)
    U=extractdata(mlp(dlarray(Xg/L),p,numel(layers)-1));
end
function rm=meanResidual(p,Xc,mu,lam,L,layers)
    rm=mean(residualField(p,Xc,mu,lam,L,layers));
end
function r=residualField(p,Xg,mu,lam,L,layers)
    Xs=dlarray(Xg/L);
    [rx,ry]=dlfeval(@equilibrium,Xs,p,mu,lam,L,numel(layers)-1);
    r=sqrt(extractdata(rx).^2+extractdata(ry).^2);
end
