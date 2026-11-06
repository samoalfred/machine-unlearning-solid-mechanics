%% PINN decomposition: does the physics penalty (a genuine PDE residual, not
% the target field) contribute, and does the influence step add anything?
% 2D linear elasticity PINN, single seed, automatic differentiation.
clear; clc; close all;
rng(701);
L=2; R=0.5; E=1.0; nu=0.33; sigma_inf=0.1;
N_coll=300; N_bnd=200; N_bad=40; layers=[2 50 50 50 2];
iters=400; ftiters=200;
uref=@(x,y) deal((sigma_inf/E)*x.*(1+0.5*R^2./(x.^2+y.^2)), -(nu*sigma_inf/E)*y.*(1+0.5*R^2./(x.^2+y.^2)));

Xc=sampleDomain(N_coll,L,R);
th=2*pi*rand(1,N_bnd/2); Xh=[R*cos(th);R*sin(th)];
Xo=outerEdges(N_bnd-N_bnd/2,L); Xb=[Xh,Xo];
[ubx,uby]=uref(Xb(1,:),Xb(2,:)); Ub=[ubx;uby];
badIdx=randperm(size(Xh,2),N_bad); Ub_bad=Ub;
Ub_bad(1,badIdx)=Ub_bad(1,badIdx)+0.30; Ub_bad(2,badIdx)=Ub_bad(2,badIdx)+0.20;
Xt=sampleDomain(1500,L,R); [utx,uty]=uref(Xt(1,:),Xt(2,:)); Ut=[utx;uty];

beta=3e-5;
pC=trainPINN(layers,Xc,Xb,Ub,E,nu,L,iters,beta);        % clean/retrained
pB=trainPINN(layers,Xc,Xb,Ub_bad,E,nu,L,iters,beta);    % biased

keep=true(1,size(Xb,2)); keep(badIdx)=false;
Xbk=Xb(:,keep); Ubk=Ub(:,keep);

% (1) warm-FT WITH physics, no influence step
p1=fineTune(pB,layers,Xc,Xbk,Ubk,E,nu,L,ftiters,beta);
% (2) warm-FT WITHOUT physics (beta=0), no influence step
p2=fineTune(pB,layers,Xc,Xbk,Ubk,E,nu,L,ftiters,0);
% (3) influence step + warm-FT WITH physics (full method)
pInf=influenceStep(pB,Xb(:,badIdx),Ub_bad(:,badIdx),layers,5e-4);
p3=fineTune(pInf,layers,Xc,Xbk,Ubk,E,nu,L,ftiters,beta);

nm={'Clean/Retrained','Biased','warmFT +physics','warmFT no-physics','full (infl+phys)'};
ps={pC,pB,p1,p2,p3};
fprintf('\n=== PINN decomposition (single seed) ===\n');
fprintf('%-20s  MSE-to-ref     mean|div sigma|\n','Condition');
for i=1:numel(ps)
    m=predMSE(ps{i},Xt,Ut,layers,L);
    r=meanResidual(ps{i},Xc,E,nu,L,layers);
    fprintf('%-20s  %.3e      %.3f\n', nm{i}, m, r);
end

%% helpers
function X=sampleDomain(N,L,R); X=zeros(2,N); c=0;
    while c<N, p=-L+2*L*rand(2,1); if norm(p)>R, c=c+1; X(:,c)=p; end, end
end
function X=outerEdges(N,L); X=zeros(2,N);
    for i=1:N, s=randi(4); t=-L+2*L*rand();
        switch s, case 1,X(:,i)=[t;L]; case 2,X(:,i)=[t;-L]; case 3,X(:,i)=[L;t]; case 4,X(:,i)=[-L;t]; end
    end
end
function Y=mlp(X,p,L); A=X;
    for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L));
end
function p=initParams(layers); p=struct;
    for i=1:numel(layers)-1
        p.(sprintf('W%d',i))=dlarray(sqrt(2/layers(i))*randn(layers(i+1),layers(i)));
        p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1));
    end
end
function p=trainPINN(layers,Xc,Xb,Ub,E,nu,L,iters,beta)
    p=initParams(layers); Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub);
    aG=[]; aS=[]; lr=2e-3;
    for it=1:iters
        [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,E,nu,L,numel(layers)-1,beta);
        if it==round(0.7*iters), lr=5e-4; end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr);
    end
end
function p=fineTune(p0,layers,Xc,Xb,Ub,E,nu,L,iters,beta)
    p=p0; Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub);
    aG=[]; aS=[]; lr=5e-4;
    for it=1:iters
        [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,E,nu,L,numel(layers)-1,beta);
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr);
    end
end
function p=influenceStep(pB,Xbad,Ubad,layers,al)
    nL=numel(layers)-1;
    Xbd=dlarray(Xbad/2); Ubd=dlarray(Ubad);         % L=2 normalization
    [~,g]=dlfeval(@(pp) dOnly(pp,Xbd,Ubd,nL),pB);
    p=pB;
    for i=1:nL
        p.(sprintf('W%d',i))=p.(sprintf('W%d',i))+al*g.(sprintf('W%d',i));
        p.(sprintf('b%d',i))=p.(sprintf('b%d',i))+al*g.(sprintf('b%d',i));
    end
end
function [loss,grad]=dOnly(p,Xb,Ub,nL)
    loss=mean((mlp(Xb,p,nL)-Ub).^2,'all'); grad=dlgradient(loss,p);
end
function [loss,grad]=loss(p,Xc,Xb,Ub,E,nu,L,nL,beta)
    dataLoss=mean((mlp(Xb,p,nL)-Ub).^2,'all');
    if beta>0
        [rx,ry]=equilibrium(Xc,p,E,nu,L,nL);
        loss=dataLoss+beta*mean(rx.^2+ry.^2,'all');
    else
        loss=dataLoss;
    end
    grad=dlgradient(loss,p);
end
function [rx,ry]=equilibrium(Xs,p,E,nu,L,nL)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L;
    duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    exx=dux(1,:); eyy=duy(2,:); gxy=dux(2,:)+duy(1,:);
    c=E/(1-nu^2); G=E/(2*(1+nu));
    sxx=c*(exx+nu*eyy); syy=c*(eyy+nu*exx); sxy=G*gxy;
    dsxx=dlgradient(sum(sxx,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dsyy=dlgradient(sum(syy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dsxy=dlgradient(sum(sxy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    rx=dsxx(1,:)+dsxy(2,:); ry=dsxy(1,:)+dsyy(2,:);
end
function mse=predMSE(p,Xt,Ut,layers,L)
    U=extractdata(mlp(dlarray(Xt/L),p,numel(layers)-1)); mse=mean((U(:)-Ut(:)).^2);
end
function rm=meanResidual(p,Xc,E,nu,L,layers)
    Xs=dlarray(Xc/L); [rx,ry]=dlfeval(@equilibrium,Xs,p,E,nu,L,numel(layers)-1);
    rm=mean(sqrt(extractdata(rx).^2+extractdata(ry).^2));
end
