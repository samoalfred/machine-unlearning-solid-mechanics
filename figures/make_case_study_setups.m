%% Figure 2 of the manuscript: 2x3 schematic of the six case-study setups.
% Writes Revisions/Revised Manuscript/case_study_setups.png (600 dpi).
% Palette: structure black, normal load/tension blue, shear orange (dashed),
% axes/deflection gray, corrupted data and corruption notes red.
%
% Each panel is drawn on a normalized [0,1]x[0,1] canvas using the helpers at
% the bottom: panel() makes a clean tile, arrow() draws a line with a filled head,
% cap() writes the governing relation and the corruption note, and xmark() draws
% the red crosses that mark corrupted boundary points.
clear; clc; close all;
OUT = 'case_study_setups.png';   % saved in the current folder (repository root)

K  = [0 0 0];            % structure
GY = [0.4 0.4 0.4];      % axes / deflection
LD = [0.00 0.45 0.74];   % normal load / tension
SH = [0.85 0.33 0.10];   % shear
RD = [0.80 0.08 0.08];   % corrupted points (red crosses)
NT = [0.30 0.30 0.30];   % corruption notes (dark grey)

fig = figure('Units','inches','Position',[1 1 10.2 7.4],'Color','w');
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

%% (a) Beam deflection
ax = panel(); title(ax,'(a) Beam deflection','Interpreter','latex','FontSize',14);
line(ax,[.12 .12],[.45 .80],'Color',K,'LineWidth',2);                                    % wall
for q=linspace(.47,.80,6), line(ax,[.07 .12],[q-.03 q],'Color',K,'LineWidth',.6); end   % hatching
line(ax,[.12 .84],[.68 .68],'Color',K,'LineWidth',2);                                    % beam
xs = linspace(.12,.86,60); line(ax,xs,.68-.17*((xs-.12)/.74).^2,'Color',GY,'LineWidth',1.1,'LineStyle','--');
arrow(ax,.84,.76,.84,.54,LD,1.6,'-'); text(ax,.88,.66,'$F$','Interpreter','latex','FontSize',14);
cap(ax,'$\delta = F/k$','corrupted: $\delta$ doubled $\rightarrow 2F/k$',NT);

%% (b) Temperature-dependent stiffness
ax = panel(); title(ax,'(b) Temperature-dependent','Interpreter','latex','FontSize',14);
rectangle(ax,'Position',[.40 .45 .13 .30],'EdgeColor',K,'LineWidth',1.2);
arrow(ax,.465,.75,.465,.90,LD,1.5,'-'); arrow(ax,.465,.45,.465,.30,LD,1.5,'-');
text(ax,.50,.88,'$\sigma$','Interpreter','latex','FontSize',14);
line(ax,[.68 .68],[.50 .78],'Color',K,'LineWidth',1.4);
rectangle(ax,'Position',[.652 .442 .056 .056],'Curvature',[1 1],'EdgeColor',K,'LineWidth',1.2);
text(ax,.74,.62,'$T$','Interpreter','latex','FontSize',14);
cap(ax,'$\varepsilon = \sigma\,/\,[E_0(1-\alpha_T T)]$','corrupted: $T$ sensor offset $+20^\circ$C',NT);

%% (c) Multi-axial stress
ax = panel(); title(ax,'(c) Multi-axial stress','Interpreter','latex','FontSize',14);
rectangle(ax,'Position',[.38 .42 .24 .24],'EdgeColor',K,'LineWidth',1.3);
arrow(ax,.38,.54,.20,.54,LD,1.5,'-'); arrow(ax,.62,.54,.80,.54,LD,1.5,'-');
text(ax,.12,.54,'$\sigma_x$','Interpreter','latex','FontSize',14);
arrow(ax,.44,.70,.58,.70,SH,1.5,'--'); arrow(ax,.56,.38,.42,.38,SH,1.5,'--');
text(ax,.66,.72,'$\tau_{xy}$','Interpreter','latex','FontSize',14);
cap(ax,'$\varepsilon_x=\sigma_x/E,\ \ \gamma_{xy}=\tau_{xy}/G$','corrupted: spurious $\sigma_x$--$\tau_{xy}$ coupling',NT);

%% (d) Cyclic fatigue: D on the vertical axis, N on the horizontal axis
ax = panel(); title(ax,'(d) Cyclic fatigue','Interpreter','latex','FontSize',14);
tt = linspace(0,1,200); line(ax,.22+.52*tt,.82+.06*sin(2*pi*4*tt),'Color',LD,'LineWidth',1.0);
text(ax,.77,.82,'load','Interpreter','latex','FontSize',11,'Color',GY);
line(ax,[.22 .22],[.36 .70],'Color',GY,'LineWidth',.9); line(ax,[.22 .84],[.36 .36],'Color',GY,'LineWidth',.9);
nn = linspace(0,1,120); line(ax,.22+.60*nn,.36+.28*(1-exp(-3.2*nn)),'Color',K,'LineWidth',1.4);
text(ax,.15,.68,'$D$','Interpreter','latex','FontSize',14,'HorizontalAlignment','center');
text(ax,.86,.31,'$N$','Interpreter','latex','FontSize',14);
text(ax,.53,.30,'cycles (log scale)','Interpreter','latex','FontSize',10,'Color',GY,'HorizontalAlignment','center');
cap(ax,'$D = 1-\exp(-c\,N\,\sigma^{m})$','corrupted: damage $\times 1.5$ (clipped at 1)',NT);

%% (e) Linear elasticity PINN: plate with hole, corrupted hole-boundary points
ax = panel(); title(ax,'(e) Linear elasticity PINN','Interpreter','latex','FontSize',14);
rectangle(ax,'Position',[.33 .42 .34 .34],'EdgeColor',K,'LineWidth',1.3);
th = linspace(0,2*pi,200); line(ax,.50+.075*cos(th),.59+.075*sin(th),'Color',K,'LineWidth',1.3);
arrow(ax,.33,.59,.17,.59,LD,1.5,'-'); arrow(ax,.67,.59,.83,.59,LD,1.5,'-');
rng(7); tb = sort(2*pi*rand(1,9)); xmark(ax,.50+.075*cos(tb),.59+.075*sin(tb),RD);
cap(ax,'plate with hole,\ $\nabla\!\cdot\!\sigma = 0$','corrupted: hole-boundary displacements ($\times$)',NT);

%% (f) Neo-Hookean PINN: plate stretched horizontally, hole becomes an ellipse
ax = panel(); title(ax,'(f) Hyperelasticity PINN','Interpreter','latex','FontSize',14);
rectangle(ax,'Position',[.28 .45 .44 .28],'EdgeColor',K,'LineWidth',1.3);           % stretched plate
line(ax,[.33 .67 .67 .33 .33],[.42 .42 .76 .76 .42],'Color',GY,'LineWidth',.8,'LineStyle',':');   % undeformed outline
ex = .50+.095*cos(th); ey = .59+.058*sin(th); line(ax,ex,ey,'Color',K,'LineWidth',1.3);   % elongated hole
arrow(ax,.28,.59,.12,.59,LD,2.0,'-'); arrow(ax,.72,.59,.88,.59,LD,2.0,'-');
rng(11); tb = sort(2*pi*rand(1,9)); xmark(ax,.50+.095*cos(tb),.59+.058*sin(tb),RD);
cap(ax,'finite stretch,\ $\nabla\!\cdot\!\mathbf{P} = \mathbf{0}$','corrupted: hole-boundary displacements ($\times$)',NT);

annotation(fig,'rectangle',[0.004 0.004 0.992 0.992],'Color',K,'LineWidth',1.2);   % bounding box around the whole figure
exportgraphics(fig,OUT,'Resolution',600,'BackgroundColor','white');
fprintf('saved %s\n',OUT);

%% ---- helpers ----
function ax = panel()   % clean equal-aspect [0,1]^2 canvas
    ax = nexttile; hold(ax,'on'); axis(ax,'equal'); xlim(ax,[0 1]); ylim(ax,[0 1]);
    axis(ax,'off'); set(ax,'Clipping','off');
end
function cap(ax,eq,corr,col)   % governing relation (center) and corruption note (below)
    text(ax,.5,.13,eq,'Interpreter','latex','FontSize',13,'HorizontalAlignment','center');
    text(ax,.5,.02,corr,'Interpreter','latex','FontSize',10.5,'Color',col,'HorizontalAlignment','center');
end
function arrow(ax,x0,y0,x1,y1,col,lw,ls)   % line with a filled triangular head
    line(ax,[x0 x1],[y0 y1],'Color',col,'LineWidth',lw,'LineStyle',ls);
    d=[x1-x0,y1-y0]; u=d/norm(d); p=[-u(2) u(1)]; hs=.032; tip=[x1 y1];
    b1=tip-hs*u*1.7+hs*p; b2=tip-hs*u*1.7-hs*p;
    patch(ax,[tip(1) b1(1) b2(1)],[tip(2) b1(2) b2(2)],col,'EdgeColor',col);
end
function xmark(ax,x,y,col)   % red crosses marking corrupted boundary points
    plot(ax,x,y,'x','Color',col,'MarkerSize',9,'LineWidth',1.8);
end
