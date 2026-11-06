%% Algorithm 2 on the two PINN case studies: results, figures, stats.
% Consistent pipeline: S seeds of Algorithm 2 (Newton step on the boundary
% data-loss Hessian + physics-regularized fine-tune), per-seed MSE for p/d, and
% regenerated figures (result bar chart + boundary displacements, equilibrium
% residual maps, and the displacement field for the hyperelastic case).
%
% Case studies: CS5 = 2D linear elasticity, CS6 = Neo-Hookean hyperelasticity,
% both on a square plate (half-width L) with a central circular hole (radius R).
% The corruption perturbs a subset of hole-boundary displacements. The influence
% step uses the boundary DATA-loss Hessian only (matrix-free, no physics in the
% solve); the equilibrium physics is reintroduced during the fine-tune.
%
% Resumable: checkpointed per seed to results/algorithm2_pinn_partial.mat.
clear; clc; close all;

% ---- configuration ----
L=2; R=0.5; Nc=300; Nb=200; Nbad=40; beta=3e-5;   % plate half-width, hole radius, #collocation/#boundary/#corrupted points, physics weight
S=5; iters=180; ft=150; cgIters=50; gammaFrac=1e-3; DPI=200; NG=70; seedsPerCall=5;   % seeds, train/fine-tune/CG iters, damping fraction, fig DPI, map grid, seeds per run

% ---- resume from checkpoint (or start fresh) ----
if ~exist('results','dir'); mkdir('results'); end
pf=fullfile('results','algorithm2_pinn_partial.mat');   % per-seed checkpoint file
if exist(pf,'file'), Ll=load(pf); Rr=Ll.Rr; acc=Ll.acc;   % Rr = finalized results; acc = current-case accumulator
else, Rr=struct('name',{},'biased',{},'alg2',{},'retrained',{},'biased_s',{},'alg2_s',{},'retr_s',{},'errred',{},'pval',{},'cohend',{},'res',{});
    acc=struct('p',0,'mb',[],'ma',[],'mr',[],'K',[],'res',[]); end

% ---- pick the next PINN case to work on ----
pidx=numel(Rr)+1;   % 1 = linear elasticity (CS5), 2 = Neo-Hookean (CS6)
if pidx>2
    % both cases done: print the summary table, save, and stop
    fprintf('\n=== Algorithm 2 PINN summary (%d seeds) ===\n',S);
    for q=1:2, fprintf('CS%d %-26s biased %.2e alg2 %.2e retrain %.2e | ErrRed %.1f%% p=%.2e d=%.2f res %.3f/%.3f/%.3f\n', ...
        q+4,Rr(q).name,Rr(q).biased,Rr(q).alg2,Rr(q).retrained,Rr(q).errred,Rr(q).pval,Rr(q).cohend,Rr(q).res); end
    save(fullfile('results','algorithm2_pinn.mat'),'Rr'); return;
end
if acc.p~=pidx, acc=struct('p',pidx,'mb',[],'ma',[],'mr',[],'K',[],'res',[]); end   % reset accumulator if it belongs to a different case
% ---- Reference displacement field (uref) ----
% Both PINN case studies generate their boundary data and test field from the
% closed-form reference field
%     u_x =  a*x*(1 + R^2/(2 r^2)),   u_y = -b*y*(1 + R^2/(2 r^2)),   r^2 = x^2 + y^2,
% a uniform-tension field (with Poisson contraction for Case Study 5) modulated by
% a term that decays away from the hole of radius R = 0.5:
%     Case Study 5 (linear elasticity): a = load/E = 0.1,  b = nu*load/E = 0.033
%     Case Study 6 (Neo-Hookean):       a = b = load = 0.05
% This field resembles, but is not, the exact plate-with-hole (Kirsch) solution,
% and it does not satisfy the equilibrium equations exactly. All PINN errors
% (clean, biased, unlearned, retrained) are measured against this same reference,
% so the comparisons between models are unaffected.
isH=(pidx==2);   % isH = true for the Neo-Hookean hyperelastic case
if isH, layers=[2 30 30 30 2]; mu=1; lam=1; la=0.05; off=[0.15 0.10]; nm='Neo-Hookean Hyperelasticity';   % network, material constants, load amplitude, corruption offset
    uref=@(x,y) deal(la*x.*(1+0.5*R^2./(x.^2+y.^2)), -la*y.*(1+0.5*R^2./(x.^2+y.^2)));   % analytic reference displacement field
else, layers=[2 50 50 50 2]; E=1; nu=0.33; la=0.1; off=[0.30 0.20]; nm='2D Linear Elasticity';   % linear elasticity: wider net + its own constants
    uref=@(x,y) deal((la/E)*x.*(1+0.5*R^2./(x.^2+y.^2)), -(nu*la/E)*y.*(1+0.5*R^2./(x.^2+y.^2))); mu=E; lam=nu;   % reuse mu/lam slots for E/nu
end
nL=numel(layers)-1; ranThis=0;   % nL = number of weight layers; ranThis = seeds done this call

% ---- per-seed loop (runs up to seedsPerCall seeds, then checkpoints) ----
while numel(acc.mb)<S && ranThis<seedsPerCall
    s=numel(acc.mb)+1; ranThis=ranThis+1; rng(60+s+10*pidx);   % deterministic per-case, per-seed RNG
    Xc=sampleDomain(Nc,L,R); th=2*pi*rand(1,Nb/2); Xh=[R*cos(th);R*sin(th)];   % collocation pts; hole-rim boundary pts
    Xo=outerEdges(Nb-Nb/2,L); Xb=[Xh,Xo]; [a,b]=uref(Xb(1,:),Xb(2,:)); Ub=[a;b];   % outer-edge pts; clean boundary displacements
    bad=randperm(size(Xh,2),Nbad); Ubd=Ub; Ubd(1,bad)=Ubd(1,bad)+off(1); Ubd(2,bad)=Ubd(2,bad)+off(2);   % corrupt a subset of hole-rim displacements
    Xt=sampleDomain(1500,L,R); [tx,ty]=uref(Xt(1,:),Xt(2,:)); Ut=[tx;ty]; keep=true(1,size(Xb,2)); keep(bad)=false;   % test set; keep-mask of retained boundary pts
    pC=trainP(layers,Xc,Xb,Ub,L,iters,beta,mu,lam,isH); pB=trainP(layers,Xc,Xb,Ubd,L,iters,beta,mu,lam,isH);   % pC = retrained (clean); pB = biased (corrupted)
    gbad=dlfeval(@gradSub,pB,dlarray(Xb(:,bad)/L),dlarray(Ubd(:,bad)),Nb,nL);   % g_bad: corrupted pts' share of the boundary data-loss gradient
    lmax=powerLambda(pB,dlarray(Xb/L),dlarray(Ubd),nL,20); gamma=gammaFrac*lmax;   % top eigenvalue of the boundary-data Hessian, then damping
    v=cg(pB,gbad,dlarray(Xb/L),dlarray(Ubd),gamma,cgIters,1e-6,nL);   % Newton direction on the data-loss Hessian (physics excluded here)
    pN=axpy(pB,1.0,v); pA=fineTune(pN,layers,Xc,Xb(:,keep),Ub(:,keep),L,ft,beta,mu,lam,isH);   % influence step, then data+physics fine-tune on retained pts
    acc.mb(end+1)=pmse(pB,Xt,Ut,nL,L); acc.ma(end+1)=pmse(pA,Xt,Ut,nL,L); acc.mr(end+1)=pmse(pC,Xt,Ut,nL,L);   % record biased/unlearned/retrained test MSE
    if s==1, acc.K=struct('pC',pC,'pB',pB,'pA',pA,'Xc',Xc,'Xb',Xb,'Xh',Xh,'bad',bad,'Ubd',Ubd,'mu',mu,'lam',lam,'isH',isH,'layers',layers);   % keep first seed for figures
        acc.res=[meanRes(pC,Xc,mu,lam,L,nL,isH); meanRes(pB,Xc,mu,lam,L,nL,isH); meanRes(pA,Xc,mu,lam,L,nL,isH)]'; end   % mean equilibrium residual, 3 models
    save(pf,'Rr','acc');   % checkpoint after every seed (interruptible)
    fprintf('CS%d seed %d/%d: biased %.3e alg2 %.3e retrain %.3e\n',pidx+4,s,S,acc.mb(end),acc.ma(end),acc.mr(end));
end

% ---- finalize the case once all S seeds are done ----
if numel(acc.mb)>=S
    mb=acc.mb(:); ma=acc.ma(:); mr=acc.mr(:); [~,pv]=ttest(mb,ma); dcoh=(mean(mb)-mean(ma))/std(mb-ma);   % paired t-test + Cohen's d (biased vs unlearned)
    Rr(pidx)=struct('name',nm,'biased',mean(mb),'alg2',mean(ma),'retrained',mean(mr), ...
        'biased_s',std(mb),'alg2_s',std(ma),'retr_s',std(mr),'errred',(mean(mb)-mean(ma))/mean(mb)*100,'pval',pv,'cohend',dcoh,'res',acc.res);   % store means, std devs, % error reduction, stats, residuals
    makePINNfigs(pidx,nm,acc.K,nL,L,R,NG,DPI,[mean(mr) mean(mb) mean(ma)],[std(mr) std(mb) std(ma)]);
    acc=struct('p',0,'mb',[],'ma',[],'mr',[],'K',[],'res',[]); save(pf,'Rr','acc');   % clear accumulator so next run starts the next case
    fprintf('  finalized CS%d (p=%.2e d=%.2f). Re-run for next.\n',pidx+4,pv,dcoh);
    return;
else
    fprintf('CS%d: %d/%d seeds done; re-run.\n',pidx+4,numel(acc.mb),S); return;
end
%% ---- figures ----
% Builds three figures: (1) boundary data + MSE bar chart, (2) equilibrium-residual
% maps (clean/biased/unlearned), (3) for the hyperelastic case, displacement fields.
function makePINNfigs(pidx,nm,K,nL,L,R,NG,DPI,barMean,barStd)
    th2=linspace(0,2*pi,200); pfx=sprintf('problem%d',pidx+4);
    % result figure: boundary u_x + bar chart
    figure('Position',[60 60 1100 420],'Color','w');
    subplot(1,2,1); hold on; grid on;
    scatter(K.Xb(1,:),K.Ubd(1,:),25,'ko','filled');
    scatter(K.Xh(1,K.bad),K.Ubd(1,K.bad),45,'rx','LineWidth',1.5);
    xlabel('x'); ylabel('u_x (boundary data)'); title('(a) Boundary data'); legend('Clean','Corrupted','Location','best');
    subplot(1,2,2); bar(barMean); hold on; errorbar(1:3,barMean,barStd,'k.','LineWidth',1.5);
    set(gca,'XTickLabel',{'Clean','Biased','Unlearned'}); ylabel('MSE (mean over seeds)'); title('(b) Performance (Algorithm 2)'); grid on;
    sgtitle(sprintf('Case Study %d: %s PINN (Algorithm 2)',pidx+4,nm),'FontSize',13);
    exportgraphics(gcf,[pfx '_results.png'],'Resolution',DPI); fprintf('  saved %s_results.png\n',pfx);
    % residual maps
    gx=linspace(-L,L,NG); [Gx,Gy]=meshgrid(gx,gx); m=sqrt(Gx.^2+Gy.^2)>R; Xg=[Gx(m)';Gy(m)'];   % grid outside the hole
    Rc=resField(K.pC,Xg,K.mu,K.lam,L,nL,K.isH); Rb=resField(K.pB,Xg,K.mu,K.lam,L,nL,K.isH); Ru=resField(K.pA,Xg,K.mu,K.lam,L,nL,K.isH);
    figure('Position',[60 60 1200 400],'Color','w'); cl=[0 max(Rb)]; Rs={Rc,Rb,Ru}; tt={'(a) Clean','(b) Biased','(c) Unlearned'};
    if K.isH, lbl='|\nabla\cdot P|'; else, lbl='|\nabla\cdot\sigma|'; end
    for k=1:3, subplot(1,3,k); Z=nan(size(Gx)); Z(m)=Rs{k};
        imagesc(gx,gx,Z); set(gca,'YDir','normal'); hold on; plot(R*cos(th2),R*sin(th2),'w-','LineWidth',2);
        axis equal; axis tight; caxis(cl); c=colorbar; c.Label.String=lbl; title(tt{k}); xlabel('x'); ylabel('y'); end
    sgtitle(sprintf('Equilibrium residual %s (Algorithm 2, computed by AD)',lbl),'FontSize',13);
    exportgraphics(gcf,[pfx '_residuals.png'],'Resolution',DPI); fprintf('  saved %s_residuals.png\n',pfx);
    % displacement field for hyperelastic
    if K.isH
        UC=extractdata(mlp(dlarray(Xg/L),K.pC,nL)); UB=extractdata(mlp(dlarray(Xg/L),K.pB,nL)); UU=extractdata(mlp(dlarray(Xg/L),K.pA,nL));
        mag=@(U) hypot(U(1,:),U(2,:)); mags={mag(UC),mag(UB),mag(UU),abs(mag(UB)-mag(UC))}; ttl={'(a) Clean','(b) Biased','(c) Unlearned','(d) |Biased-Clean|'};
        figure('Position',[60 60 1100 800],'Color','w');
        for k=1:4, subplot(2,2,k); Z=nan(size(Gx)); Z(m)=mags{k};
            imagesc(gx,gx,Z); set(gca,'YDir','normal'); hold on; plot(R*cos(th2),R*sin(th2),'w-','LineWidth',2);
            axis equal; axis tight; colorbar; title(ttl{k}); xlabel('x'); ylabel('y'); end
        sgtitle('Displacement magnitude (Neo-Hookean PINN, Algorithm 2)','FontSize',13);
        exportgraphics(gcf,[pfx '_displacement.png'],'Resolution',DPI); fprintf('  saved %s_displacement.png\n',pfx);
    end
end

%% ---- helpers ----
% mlp: forward pass -- tanh hidden layers, linear output
function Y=mlp(X,p,L); A=X; for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L)); end
% initP: He-style init; for PINNs the last layer starts near zero (zl) for a gentle start
function p=initP(layers,zl); p=struct; nL=numel(layers)-1;
    for i=1:nL, sc=sqrt(2/layers(i)); if zl&&i==nL, sc=0.01; end
        p.(sprintf('W%d',i))=dlarray(sc*randn(layers(i+1),layers(i))); p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1)); end
end
% trainP: PINN training with Adam, minimizing data loss + beta*physics loss
function p=trainP(layers,Xc,Xb,Ub,L,iters,beta,mu,lam,H)
    p=initP(layers,H); Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub); aG=[];aS=[];lr=2e-3;nL=numel(layers)-1;   % inputs scaled by L
    for it=1:iters, [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,L,nL,beta,mu,lam,H); if it==round(0.7*iters),lr=5e-4;end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
% fineTune: like trainP but starts from the given params (used after the influence step)
function p=fineTune(p,layers,Xc,Xb,Ub,L,ft,beta,mu,lam,H)
    Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub); aG=[];aS=[];lr=5e-4;nL=numel(layers)-1;
    for it=1:ft, [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,L,nL,beta,mu,lam,H); [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
% loss: boundary data MSE + beta * mean squared equilibrium residual
function [l,grad]=loss(p,Xc,Xb,Ub,L,nL,beta,mu,lam,H)
    dl=mean((mlp(Xb,p,nL)-Ub).^2,'all');   % boundary data term
    if H, [rx,ry]=eqH(Xc,p,L,nL,mu,lam); else, [rx,ry]=eqL(Xc,p,L,nL,mu,lam); end   % equilibrium residual (Neo-Hookean or linear)
    l=dl+beta*mean(rx.^2+ry.^2,'all'); grad=dlgradient(l,p);
end
% gradSub: g_bad = gradient of (sum of subset squared errors)/Nb (boundary data loss)
function g=gradSub(p,Xs,Ts,Nb,nL); l=sum((mlp(Xs,p,nL)-Ts).^2,'all')/Nb; g=dlgradient(l,p); end
% hvp: Hessian-vector product of the boundary DATA loss, via double backprop (no H formed)
function Hv=hvp(p,v,Xb,Ub,nL)
    g=dlgradient(mean((mlp(Xb,p,nL)-Ub).^2,'all'),p,'EnableHigherDerivatives',true);
    gv=0; f=fieldnames(g); for i=1:numel(f), gv=gv+sum(g.(f{i}).*v.(f{i}),'all'); end; Hv=dlgradient(gv,p);
end
% powerLambda: power iteration (through hvp) for the top data-Hessian eigenvalue
function lam=powerLambda(p,Xb,Ub,nL,it0); v=randLike(p); v=nrm(v);
    for it=1:it0, Hv=dlfeval(@hvp,p,v,Xb,Ub,nL); lam=sdot(v,Hv); v=nrm(Hv); end
end
% cg: damped conjugate gradient for (H_data + lam I) v = g, with a curvature guard
function v=cg(p,g,Xb,Ub,lam,maxit,tol,nL)
    x=scal(g,0); r=g; pp=r; rs=sdot(r,r); g0=sqrt(rs);
    for it=1:maxit, Hp=dlfeval(@hvp,p,pp,Xb,Ub,nL); Ap=axpy(Hp,lam,pp);   % Ap = (H + lam I) pp
        den=sdot(pp,Ap); if den<=0, den=den+lam*sdot(pp,pp)+1e-12; end   % guard against non-positive curvature
        al=rs/den; x=axpy(x,al,pp); r=axpy(r,-al,Ap); rs2=sdot(r,r); if sqrt(rs2)<tol*g0, break; end
        pp=axpy(r,rs2/rs,pp); rs=rs2; end
    v=x;
end
% eqL: LINEAR elasticity equilibrium residual div(sigma), via automatic differentiation
function [rx,ry]=eqL(Xs,p,L,nL,E,nu)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L; duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;   % displacement gradients
    exx=dux(1,:); eyy=duy(2,:); gxy=dux(2,:)+duy(1,:); c=E/(1-nu^2); G=E/(2*(1+nu));   % strains + plane-stress constants
    sxx=c*(exx+nu*eyy); syy=c*(eyy+nu*exx); sxy=G*gxy;   % stresses (plane-stress constitutive law)
    dsxx=dlgradient(sum(sxx,'all'),Xs,'EnableHigherDerivatives',true)/L; dsyy=dlgradient(sum(syy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dsxy=dlgradient(sum(sxy,'all'),Xs,'EnableHigherDerivatives',true)/L; rx=dsxx(1,:)+dsxy(2,:); ry=dsxy(1,:)+dsyy(2,:);   % residual = divergence of stress
end
% eqH: NEO-HOOKEAN equilibrium residual div(P), P = first Piola-Kirchhoff stress, via AD
function [rx,ry]=eqH(Xs,p,L,nL,mu,lam)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L; duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    a=1+dux(1,:); b=dux(2,:); c=duy(1,:); d=1+duy(2,:); J=a.*d-b.*c;   % deformation gradient F and its determinant J
    iTxx=d./J; iTxy=-c./J; iTyx=-b./J; iTyy=a./J; lJ=log(J);   % inverse-transpose of F and log J
    Pxx=mu*(a-iTxx)+lam*lJ.*iTxx; Pxy=mu*(b-iTxy)+lam*lJ.*iTxy; Pyx=mu*(c-iTyx)+lam*lJ.*iTyx; Pyy=mu*(d-iTyy)+lam*lJ.*iTyy;   % P = mu(F - F^-T) + lam*lnJ*F^-T
    dPxx=dlgradient(sum(Pxx,'all'),Xs,'EnableHigherDerivatives',true)/L; dPxy=dlgradient(sum(Pxy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dPyx=dlgradient(sum(Pyx,'all'),Xs,'EnableHigherDerivatives',true)/L; dPyy=dlgradient(sum(Pyy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    rx=dPxx(1,:)+dPxy(2,:); ry=dPyx(1,:)+dPyy(2,:);   % residual = divergence of P
end
% pmse: test MSE against the analytic reference field (inputs scaled by L)
function m=pmse(p,Xt,Ut,nL,L); U=extractdata(mlp(dlarray(Xt/L),p,nL)); m=mean((U(:)-Ut(:)).^2); end
% meanRes/resField: mean |residual| over points, and the per-point residual magnitude
function rm=meanRes(p,Xc,mu,lam,L,nL,H); rm=mean(resField(p,Xc,mu,lam,L,nL,H)); end
function r=resField(p,Xg,mu,lam,L,nL,H)
    Xs=dlarray(Xg/L); if H, [rx,ry]=dlfeval(@eqH,Xs,p,L,nL,mu,lam); else, [rx,ry]=dlfeval(@eqL,Xs,p,L,nL,mu,lam); end
    r=sqrt(extractdata(rx).^2+extractdata(ry).^2);
end
% sampleDomain: rejection-sample N interior points that lie outside the hole
function X=sampleDomain(N,L,R); X=zeros(2,N); c=0; while c<N, q=-L+2*L*rand(2,1); if norm(q)>R,c=c+1;X(:,c)=q;end, end, end
% outerEdges: N points sampled uniformly along the four outer edges of the plate
function X=outerEdges(N,L); X=zeros(2,N);
    for i=1:N, s=randi(4); t=-L+2*L*rand(); switch s, case 1,X(:,i)=[t;L]; case 2,X(:,i)=[t;-L]; case 3,X(:,i)=[L;t]; case 4,X(:,i)=[-L;t]; end, end
end
% struct-vector algebra: treat the parameter struct as one vector (for CG/power iteration)
function o=axpy(a,al,b); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=a.(f{i})+al*b.(f{i}); end, end   % a + al*b
function o=scal(a,c); o=a; f=fieldnames(a); for i=1:numel(f), o.(f{i})=c*a.(f{i}); end, end   % c*a
function s=sdot(a,b); s=0; f=fieldnames(a); for i=1:numel(f), s=s+sum(a.(f{i}).*b.(f{i}),'all'); end, s=double(gather(extractdata(s))); end   % inner product
function v=randLike(p); v=p; f=fieldnames(p); for i=1:numel(f), v.(f{i})=dlarray(randn(size(p.(f{i})))); end, end   % random vector shaped like p
function v=nrm(v); n=sqrt(sdot(v,v)); f=fieldnames(v); for i=1:numel(f), v.(f{i})=v.(f{i})/n; end, end   % normalize to unit length
