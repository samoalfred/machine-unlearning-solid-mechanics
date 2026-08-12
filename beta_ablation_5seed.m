%% Physics-weight (beta) ablation on the 2D linear elasticity PINN, 5 seeds.
% Per seed: train a fresh biased model (beta=3e-5), then run the unlearning
% fine-tune at each beta and record clean-field MSE and equilibrium residual.
% Reports mean +/- std across seeds. Saves results/beta_ablation_5seed.mat.
clear; clc; close all;
L=2; R=0.5; E=1.0; nu=0.33; sinf=0.1; Nc=300; Nb=200; Nbad=40; layers=[2 50 50 50 2];
iters=300; ft=200; beta_train=3e-5; betas=[0 1e-6 1e-5 1e-4 1e-3]; S=5;
uref=@(x,y) deal((sinf/E)*x.*(1+0.5*R^2./(x.^2+y.^2)), -(nu*sinf/E)*y.*(1+0.5*R^2./(x.^2+y.^2)));
MSE=zeros(S,numel(betas)); RES=zeros(S,numel(betas));
for s=1:S
    rng(700+s);
    Xc=sampleDomain(Nc,L,R); th=2*pi*rand(1,Nb/2); Xh=[R*cos(th);R*sin(th)];
    Xo=outerEdges(Nb-Nb/2,L); Xb=[Xh,Xo]; [a,b]=uref(Xb(1,:),Xb(2,:)); Ub=[a;b];
    badIdx=randperm(size(Xh,2),Nbad); Ubd=Ub; Ubd(1,badIdx)=Ubd(1,badIdx)+0.30; Ubd(2,badIdx)=Ubd(2,badIdx)+0.20;
    Xt=sampleDomain(1500,L,R); [tx,ty]=uref(Xt(1,:),Xt(2,:)); Ut=[tx;ty];
    pB=trainP(layers,Xc,Xb,Ubd,L,iters,beta_train,E,nu);
    keep=true(1,size(Xb,2)); keep(badIdx)=false;
    for i=1:numel(betas)
        pU=ftP(pB,layers,Xc,Xb(:,keep),Ub(:,keep),L,ft,betas(i),E,nu);
        MSE(s,i)=predMSE(pU,Xt,Ut,layers,L); RES(s,i)=meanRes(pU,Xc,L,layers,E,nu);
    end
    fprintf('seed %d done\n',s);
end
if ~exist('results','dir'); mkdir('results'); end
ba=struct('betas',betas,'MSE',MSE,'RES',RES); save(fullfile('results','beta_ablation_5seed.mat'),'ba');
fprintf('\n=== beta ablation (2D elasticity PINN, %d seeds) mean +/- std ===\n',S);
fprintf('%-10s %-22s %-22s\n','beta','MSE-to-ref','mean|div sigma|');
for i=1:numel(betas)
    fprintf('%-10.0e %.3e +/- %.1e   %.3f +/- %.3f\n',betas(i), ...
        mean(MSE(:,i)),std(MSE(:,i)), mean(RES(:,i)),std(RES(:,i)));
end
fprintf('Saved results/beta_ablation_5seed.mat\n');

%% helpers
function Y=mlp(X,p,L); A=X; for i=1:L-1, A=tanh(p.(sprintf('W%d',i))*A+p.(sprintf('b%d',i))); end
    Y=p.(sprintf('W%d',L))*A+p.(sprintf('b%d',L)); end
function p=initP(layers); p=struct;
    for i=1:numel(layers)-1, p.(sprintf('W%d',i))=dlarray(sqrt(2/layers(i))*randn(layers(i+1),layers(i)));
        p.(sprintf('b%d',i))=dlarray(zeros(layers(i+1),1)); end
end
function p=trainP(layers,Xc,Xb,Ub,L,iters,beta,E,nu)
    p=initP(layers); Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub); aG=[];aS=[];lr=2e-3;nL=numel(layers)-1;
    for it=1:iters, [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,L,nL,beta,E,nu); if it==round(0.7*iters),lr=5e-4;end
        [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
function p=ftP(p,layers,Xc,Xb,Ub,L,ft,beta,E,nu)
    Xcd=dlarray(Xc/L); Xbd=dlarray(Xb/L); Ubd=dlarray(Ub); aG=[];aS=[];lr=5e-4;nL=numel(layers)-1;
    for it=1:ft, [~,g]=dlfeval(@loss,p,Xcd,Xbd,Ubd,L,nL,beta,E,nu); [p,aG,aS]=adamupdate(p,g,aG,aS,it,lr); end
end
function [l,grad]=loss(p,Xc,Xb,Ub,L,nL,beta,E,nu)
    dl=mean((mlp(Xb,p,nL)-Ub).^2,'all');
    if beta>0, [rx,ry]=eqLin(Xc,p,L,nL,E,nu); l=dl+beta*mean(rx.^2+ry.^2,'all'); else, l=dl; end
    grad=dlgradient(l,p);
end
function [rx,ry]=eqLin(Xs,p,L,nL,E,nu)
    U=mlp(Xs,p,nL); ux=U(1,:); uy=U(2,:);
    dux=dlgradient(sum(ux,'all'),Xs,'EnableHigherDerivatives',true)/L; duy=dlgradient(sum(uy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    exx=dux(1,:); eyy=duy(2,:); gxy=dux(2,:)+duy(1,:); c=E/(1-nu^2); G=E/(2*(1+nu));
    sxx=c*(exx+nu*eyy); syy=c*(eyy+nu*exx); sxy=G*gxy;
    dsxx=dlgradient(sum(sxx,'all'),Xs,'EnableHigherDerivatives',true)/L; dsyy=dlgradient(sum(syy,'all'),Xs,'EnableHigherDerivatives',true)/L;
    dsxy=dlgradient(sum(sxy,'all'),Xs,'EnableHigherDerivatives',true)/L; rx=dsxx(1,:)+dsxy(2,:); ry=dsxy(1,:)+dsyy(2,:);
end
function m=predMSE(p,Xt,Ut,layers,L); U=extractdata(mlp(dlarray(Xt/L),p,numel(layers)-1)); m=mean((U(:)-Ut(:)).^2); end
function rm=meanRes(p,Xc,L,layers,E,nu)
    Xs=dlarray(Xc/L); [rx,ry]=dlfeval(@eqLin,Xs,p,L,numel(layers)-1,E,nu); rm=mean(sqrt(extractdata(rx).^2+extractdata(ry).^2));
end
function X=sampleDomain(N,L,R); X=zeros(2,N); c=0; while c<N, q=-L+2*L*rand(2,1); if norm(q)>R,c=c+1;X(:,c)=q;end, end, end
function X=outerEdges(N,L); X=zeros(2,N);
    for i=1:N, s=randi(4); t=-L+2*L*rand(); switch s, case 1,X(:,i)=[t;L]; case 2,X(:,i)=[t;-L]; case 3,X(:,i)=[L;t]; case 4,X(:,i)=[-L;t]; end, end
end
