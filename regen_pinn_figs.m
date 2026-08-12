%% Regenerate PINN heatmap figures at higher resolution (100x100 grid, 200 dpi).
% Produces problem5_residuals.png, problem6_displacement.png,
% problem6_residuals.png from freshly trained single-seed models. Does NOT
% touch results/*.mat (figures are illustrative single-seed fields).
clear; clc; close all;
NG=100; DPI=200;

%% ---------- Problem 5: 2D linear elasticity PINN ----------
rng(701);
L=2; R=0.5; E=1.0; nu=0.33; sinf=0.1; Nc=300; Nb=200; Nbad=40; layers=[2 50 50 50 2];
iters=350; ft=150; beta=3e-5;
uref=@(x,y) deal((sinf/E)*x.*(1+0.5*R^2./(x.^2+y.^2)), -(nu*sinf/E)*y.*(1+0.5*R^2./(x.^2+y.^2)));
[pC,pB,pU,~,~,~,~]=trainTrio(@eqLin,layers,L,R,Nc,Nb,Nbad,iters,ft,beta,E,nu,uref,[0.30 0.20]);

gx=linspace(-L,L,NG); [Gx,Gy]=meshgrid(gx,gx); m=sqrt(Gx.^2+Gy.^2)>R; Xg=[Gx(m)';Gy(m)'];
th2=linspace(0,2*pi,200);
Rc=resField(@eqLin,pC,Xg,L,layers,E,nu); Rb=resField(@eqLin,pB,Xg,L,layers,E,nu); Ru=resField(@eqLin,pU,Xg,L,layers,E,nu);
figure('Position',[50 50 1200 400],'Color','w'); cl=[0 max(Rb)]; Rs={Rc,Rb,Ru}; tt={'(a) Clean','(b) Biased','(c) Unlearned'};
for kk=1:3
    subplot(1,3,kk); Z=nan(size(Gx)); Z(m)=Rs{kk};
    imagesc(gx,gx,Z); set(gca,'YDir','normal'); hold on; plot(R*cos(th2),R*sin(th2),'w-','LineWidth',2);
    axis equal; axis tight; caxis(cl); c=colorbar; c.Label.String='|div \sigma|'; title(tt{kk}); xlabel('x'); ylabel('y');
end
sgtitle('PDE equilibrium residual |\nabla\cdot\sigma| (2D elasticity PINN, computed by AD)','FontSize',13);
exportgraphics(gcf,'problem5_residuals.png','Resolution',DPI); fprintf('problem5_residuals.png (%dx%d, %d dpi)\n',NG,NG,DPI);

%% ---------- Problem 6: Neo-Hookean hyperelasticity PINN ----------
rng(61);
mu=1.0; lam=1.0; la=0.05; layers6=[2 30 30 30 2]; iters6=250; ft6=100;
uref6=@(x,y) deal(la*x.*(1+0.5*R^2./(x.^2+y.^2)), -la*y.*(1+0.5*R^2./(x.^2+y.^2)));
[qC,qB,qU,~,~,~,~]=trainTrioH(layers6,L,R,Nc,Nb,Nbad,iters6,ft6,beta,mu,lam,uref6,[0.15 0.10]);

gx=linspace(-L,L,NG); [Gx,Gy]=meshgrid(gx,gx); m=sqrt(Gx.^2+Gy.^2)>R; Xg=[Gx(m)';Gy(m)'];
% displacement magnitude
UC=extractdata(mlp(dlarray(Xg/L),qC,numel(layers6)-1));
UB=extractdata(mlp(dlarray(Xg/L),qB,numel(layers6)-1));
UU=extractdata(mlp(dlarray(Xg/L),qU,numel(layers6)-1));
mag=@(U) hypot(U(1,:),U(2,:)); mags={mag(UC),mag(UB),mag(UU),abs(mag(UB)-mag(UC))};
ttl={'(a) Clean','(b) Biased','(c) Unlearned','(d) |Biased-Clean| error'};
figure('Position',[50 50 1100 800],'Color','w');
for kk=1:4
    subplot(2,2,kk); Z=nan(size(Gx)); Z(m)=mags{kk};
    imagesc(gx,gx,Z); set(gca,'YDir','normal'); hold on; plot(R*cos(th2),R*sin(th2),'w-','LineWidth',2);
    axis equal; axis tight; colorbar; title(ttl{kk}); xlabel('x'); ylabel('y');
end
sgtitle('Displacement magnitude (Neo-Hookean PINN)','FontSize',13);
exportgraphics(gcf,'problem6_displacement.png','Resolution',DPI); fprintf('problem6_displacement.png\n');

Rc=resFieldH(qC,Xg,L,layers6,mu,lam); Rb=resFieldH(qB,Xg,L,layers6,mu,lam); Ru=resFieldH(qU,Xg,L,layers6,mu,lam);
figure('Position',[50 50 1200 400],'Color','w'); cl=[0 max(Rb)]; Rs={Rc,Rb,Ru}; tt={'(a) Clean','(b) Biased','(c) Unlearned'};
for kk=1:3
    subplot(1,3,kk); Z=nan(size(Gx)); Z(m)=Rs{kk};
    imagesc(gx,gx,Z); set(gca,'YDir','normal'); hold on; plot(R*cos(th2),R*sin(th2),'w-','LineWidth',2);
    axis equal; axis tight; caxis(cl); c=colorbar; c.Label.String='|div P|'; title(tt{kk}); xlabel('x'); ylabel('y');
end
sgtitle('PDE equilibrium residual |\nabla\cdot P| (Neo-Hookean PINN, computed by AD)','FontSize',13);
exportgraphics(gcf,'problem6_residuals.png','Resolution',DPI); fprintf('problem6_residuals.png\n');
fprintf('Done.\n');

%% ================= helpers =================
function [pC,pB,pU,Xc,Xb,Xh,badIdx]=trainTrio(eqfun,layers,L,R,Nc,Nb,Nbad,iters,ft,beta,E,nu,uref,off)
    Xc=sampleDomain(Nc,L,R); th=2*pi*rand(1,Nb/2); Xh=[R*cos(th);R*sin(th)];
    Xo=outerEdges(Nb-Nb/2,L); Xb=[Xh,Xo]; [a,b]=uref(Xb(1,:),Xb(2,:)); Ub=[a;b];
    badIdx=randperm(size(Xh,2),Nbad); Ubd=Ub; Ubd(1,badIdx)=Ubd(1,badIdx)+off(1); Ubd(2,badIdx)=Ubd(2,badIdx)+off(2);
    pC=trainP(eqfun,layers,Xc,Xb,Ub,L,iters,beta,E,nu);
    pB=trainP(eqfun,layers,Xc,Xb,Ubd,L,iters,beta,E,nu);
    keep=true(1,size(Xb,2)); keep(badIdx)=false;
    pU=ftP(eqfun,pB,layers,Xc,Xb(:,keep),Ub(:,keep),L,ft,beta,E,nu);
end
function [pC,pB,pU,Xc,Xb,Xh,badIdx]=trainTrioH(layers,L,R,Nc,Nb,Nbad,iters,ft,beta,mu,lam,uref,off)
    Xc=sampleDomain(Nc,L,R); th=2*pi*rand(1,Nb/2); Xh=[R*cos(th);R*sin(th)];
    Xo=outerEdges(Nb-Nb/2,L); Xb=[Xh,Xo]; [a,b]=uref(Xb(1,:),Xb(2,:)); Ub=[a;b];
    badIdx=randperm(size(Xh,2),Nbad); Ubd=Ub; Ubd(1,badIdx)=Ubd(1,badIdx)+off(1); Ubd(2,badIdx)=Ubd(2,badIdx)+off(2);
    pC=trainPH(layers,Xc,Xb,Ub,L,iters,beta,mu,lam);
    pB=trainPH(layers,Xc,Xb,Ubd,L,iters,beta,mu,lam);
    keep=true(1,size(Xb,2)); keep(badIdx)=false;
    pU=ftPH(pB,layers,Xc,Xb(:,keep),Ub(:,keep),L,ft,beta,mu,lam);
end
function Y=mlp(X,p,L); A=X; for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L)); end
function p=initP(layers,zeroLast)
    p=struct; nL=numel(layers)-1;
    for i=1:nL, s=sqrt(2/layers(i)); if zeroLast&&i==nL, s=0.01; end
        p.(sprintf('W%d',i))=dlarray(s*randn(layers(i+1),layers(i))); p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1)); end
end
function p=trainP(eqfun,layers,Xc,Xb,Ub,L,iters,beta,E,nu)
    p=initP(layers,false); Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub); aG=[];aS=[];lr=2e-3;nL=numel(layers)-1;
    for it=1:iters, [~,g]=dlfeval(@lossL,eqfun,p,Xcd,Xbd,Ubd,L,nL,beta,E,nu); if it==round(0.7*iters),lr=5e-4;end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
function p=ftP(eqfun,p,layers,Xc,Xb,Ub,L,ft,beta,E,nu)
    Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub); aG=[];aS=[];lr=5e-4;nL=numel(layers)-1;
    for it=1:ft, [~,g]=dlfeval(@lossL,eqfun,p,Xcd,Xbd,Ubd,L,nL,beta,E,nu); [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
function [l,grad]=lossL(eqfun,p,Xc,Xb,Ub,L,nL,beta,E,nu)
    dl=mean((mlp(Xb,p,nL)-Ub).^2,'all'); [rx,ry]=eqfun(Xc,p,L,nL,E,nu); l=dl+beta*mean(rx.^2+ry.^2,'all'); grad=dlgradient(l,p);
end
function p=trainPH(layers,Xc,Xb,Ub,L,iters,beta,mu,lam)
    p=initP(layers,true); Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub); aG=[];aS=[];lr=2e-3;nL=numel(layers)-1;
    for it=1:iters, [~,g]=dlfeval(@lossH,p,Xcd,Xbd,Ubd,L,nL,beta,mu,lam); if it==round(0.7*iters),lr=5e-4;end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
function p=ftPH(p,layers,Xc,Xb,Ub,L,ft,beta,mu,lam)
    Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub); aG=[];aS=[];lr=5e-4;nL=numel(layers)-1;
    for it=1:ft, [~,g]=dlfeval(@lossH,p,Xcd,Xbd,Ubd,L,nL,beta,mu,lam); [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
function [l,grad]=lossH(p,Xc,Xb,Ub,L,nL,beta,mu,lam)
    dl=mean((mlp(Xb,p,nL)-Ub).^2,'all'); [rx,ry]=eqHyper(Xc,p,L,nL,mu,lam); l=dl+beta*mean(rx.^2+ry.^2,'all'); grad=dlgradient(l,p);
end
function [rx,ry]=eqLin(Xs,p,L,nL,E,nu)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L; duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    exx=dux(1,:); eyy=duy(2,:); gxy=dux(2,:)+duy(1,:); c=E/(1-nu^2); G=E/(2*(1+nu));
    sxx=c*(exx+nu*eyy); syy=c*(eyy+nu*exx); sxy=G*gxy;
    dsxx=dlgradient(sum(sxx,'all'),Xs,'EnableHigherDerivatives',true)/L; dsyy=dlgradient(sum(syy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dsxy=dlgradient(sum(sxy,'all'),Xs,'EnableHigherDerivatives',true)/L; rx=dsxx(1,:)+dsxy(2,:); ry=dsxy(1,:)+dsyy(2,:);
end
function [rx,ry]=eqHyper(Xs,p,L,nL,mu,lam)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L; duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    a=1+dux(1,:); b=dux(2,:); c=duy(1,:); d=1+duy(2,:); J=a.*d-b.*c;
    iTxx=d./J; iTxy=-c./J; iTyx=-b./J; iTyy=a./J; lJ=log(J);
    Pxx=mu*(a-iTxx)+lam*lJ.*iTxx; Pxy=mu*(b-iTxy)+lam*lJ.*iTxy; Pyx=mu*(c-iTyx)+lam*lJ.*iTyx; Pyy=mu*(d-iTyy)+lam*lJ.*iTyy;
    dPxx=dlgradient(sum(Pxx,'all'),Xs,'EnableHigherDerivatives',true)/L; dPxy=dlgradient(sum(Pxy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dPyx=dlgradient(sum(Pyx,'all'),Xs,'EnableHigherDerivatives',true)/L; dPyy=dlgradient(sum(Pyy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    rx=dPxx(1,:)+dPxy(2,:); ry=dPyx(1,:)+dPyy(2,:);
end
function r=resField(eqfun,p,Xg,L,layers,E,nu)
    Xs=dlarray(Xg/L); [rx,ry]=dlfeval(eqfun,Xs,p,L,numel(layers)-1,E,nu); r=sqrt(extractdata(rx).^2+extractdata(ry).^2);
end
function r=resFieldH(p,Xg,L,layers,mu,lam)
    Xs=dlarray(Xg/L); [rx,ry]=dlfeval(@eqHyper,Xs,p,L,numel(layers)-1,mu,lam); r=sqrt(extractdata(rx).^2+extractdata(ry).^2);
end
function X=sampleDomain(N,L,R); X=zeros(2,N); c=0; while c<N, q=-L+2*L*rand(2,1); if norm(q)>R,c=c+1;X(:,c)=q; end, end, end
function X=outerEdges(N,L); X=zeros(2,N);
    for i=1:N, s=randi(4); t=-L+2*L*rand(); switch s, case 1,X(:,i)=[t;L]; case 2,X(:,i)=[t;-L]; case 3,X(:,i)=[L;t]; case 4,X(:,i)=[-L;t]; end, end
end
