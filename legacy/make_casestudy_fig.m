%% High-resolution 3x2 schematic of the six case-study problem setups.
% MATLAB / exportgraphics, LaTeX interpreter, to match the manuscript figures.
% Palette: structure black, normal load/tension blue, shear orange (dashed),
% deflection/axes gray. Brainstorming figure -- not inserted into the manuscript.
%
% Each panel is drawn in a normalized [0,1]x[0,1] axis using the small helpers at
% the bottom: panel() makes a clean tile, arrow() draws a line with a filled head,
% and cap() writes the governing equation and the corruption note.
clear; clc; close all;

% ---- output path ----
OUT = 'case_study_setups_draft.png';   % early draft; the published Figure 2 comes from make_case_study_setups.m

% ---- color palette ----
K   = [0 0 0];          % structure (black)
GY  = [0.4 0.4 0.4];    % deflection / axes (gray)
LD  = [0.00 0.45 0.74]; % normal load / tension (blue)
SH  = [0.85 0.33 0.10]; % shear (orange)

fig = figure('Units','inches','Position',[1 1 7.2 9.6],'Color','w');
t = tiledlayout(3,2,'Padding','compact','TileSpacing','compact');   % 3 rows x 2 cols of panels

%% (a) Beam deflection
ax = panel(); title(ax,'(a) Beam deflection','Interpreter','latex','FontSize',13);
line(ax,[.12 .12],[.45 .80],'Color',K,'LineWidth',2);   % wall
yy = linspace(.47,.80,6); for q=yy, line(ax,[.07 .12],[q-.03 q],'Color',K,'LineWidth',.6); end   % wall hatching
line(ax,[.12 .84],[.68 .68],'Color',K,'LineWidth',2);   % undeformed beam
xs = linspace(.12,.86,60); line(ax,xs,.68-.17*((xs-.12)/.74).^2,'Color',GY,'LineWidth',1.1,'LineStyle','--');   % deflected shape
arrow(ax,.84,.76,.84,.54,LD,1.6,'-');   % tip load
text(ax,.88,.66,'$F$','Interpreter','latex','FontSize',13);
cap(ax,'$\delta = F/k$','corrupted: $\delta$ doubled $\rightarrow 2F/k$',GY);

%% (b) Temperature-dependent
ax = panel(); title(ax,'(b) Temperature-dependent','Interpreter','latex','FontSize',13);
rectangle(ax,'Position',[.40 .45 .13 .30],'EdgeColor',K,'LineWidth',1.2);   % specimen
arrow(ax,.465,.75,.465,.90,LD,1.5,'-'); arrow(ax,.465,.45,.465,.30,LD,1.5,'-');   % tension top/bottom
text(ax,.50,.88,'$\sigma$','Interpreter','latex','FontSize',13);
line(ax,[.68 .68],[.50 .78],'Color',K,'LineWidth',1.4);   % thermometer stem
rectangle(ax,'Position',[.652 .442 .056 .056],'Curvature',[1 1],'EdgeColor',K,'LineWidth',1.2);   % thermometer bulb
text(ax,.74,.62,'$T$','Interpreter','latex','FontSize',13);
cap(ax,'$\varepsilon = \sigma\,/\,E_0(1-\alpha T)$','corrupted: $T$ sensor offset $+20^\circ$',GY);

%% (c) Multi-axial stress
ax = panel(); title(ax,'(c) Multi-axial stress','Interpreter','latex','FontSize',13);
rectangle(ax,'Position',[.38 .42 .24 .24],'EdgeColor',K,'LineWidth',1.3);   % element
arrow(ax,.38,.54,.20,.54,LD,1.5,'-'); arrow(ax,.62,.54,.80,.54,LD,1.5,'-');   % normal stress sigma_x
text(ax,.12,.54,'$\sigma_x$','Interpreter','latex','FontSize',13);
arrow(ax,.44,.70,.58,.70,SH,1.5,'--'); arrow(ax,.56,.38,.42,.38,SH,1.5,'--');   % shear tau_xy (dashed)
text(ax,.66,.72,'$\tau_{xy}$','Interpreter','latex','FontSize',13);
cap(ax,'$\varepsilon_x=\sigma_x/E,\ \ \gamma=\tau/G$','corrupted: spurious $\sigma$--$\tau$ coupling',GY);

%% (d) Cyclic fatigue
ax = panel(); title(ax,'(d) Cyclic fatigue','Interpreter','latex','FontSize',13);
tt = linspace(0,1,200); line(ax,.18+.52*tt,.80+.07*sin(2*pi*4*tt),'Color',LD,'LineWidth',1.0);   % oscillating load
text(ax,.73,.80,'load','Interpreter','latex','FontSize',10,'Color',GY);
line(ax,[.18 .18],[.40 .68],'Color',GY,'LineWidth',.8); line(ax,[.18 .80],[.40 .40],'Color',GY,'LineWidth',.8);   % axes
nn = linspace(0,1,120); line(ax,.18+.62*nn,.40+.26*(1-exp(-3.2*nn)),'Color',K,'LineWidth',1.4);   % damage-vs-cycles curve
text(ax,.83,.44,'$D$','Interpreter','latex','FontSize',13);
text(ax,.49,.33,'$N$ (log)','Interpreter','latex','FontSize',10,'Color',GY,'HorizontalAlignment','center');
cap(ax,'$D = 1-\exp(-c\,N\,\sigma^{m})$','corrupted: damage $\times 1.5$',GY);

%% (e) Linear elasticity PINN
ax = panel(); title(ax,'(e) Linear elasticity PINN','Interpreter','latex','FontSize',13);
rectangle(ax,'Position',[.33 .42 .34 .34],'EdgeColor',K,'LineWidth',1.3);   % plate
rectangle(ax,'Position',[.425 .515 .15 .15],'Curvature',[1 1],'EdgeColor',K,'LineWidth',1.3);   % central hole
arrow(ax,.33,.59,.17,.59,LD,1.5,'-'); arrow(ax,.67,.59,.83,.59,LD,1.5,'-');   % far-field tension
cap(ax,'plate with hole,\ $\nabla\!\cdot\!\sigma = 0$','corrupted: faulty boundary sensor',GY);

%% (f) Hyperelasticity PINN
ax = panel(); title(ax,'(f) Hyperelasticity PINN','Interpreter','latex','FontSize',13);
patch(ax,[.36 .70 .74 .40],[.44 .42 .76 .78],'w','EdgeColor',K,'LineWidth',1.3);   % deformed plate
th = linspace(0,2*pi,100); ex=.55+.07*cos(th)*cosd(4)-.0475*sin(th)*sind(4);   % rotated/stretched hole (x)
ey=.60+.07*cos(th)*sind(4)+.0475*sin(th)*cosd(4); line(ax,ex,ey,'Color',K,'LineWidth',1.3);   % rotated/stretched hole (y)
arrow(ax,.35,.60,.17,.60,LD,2.0,'-'); arrow(ax,.75,.60,.90,.60,LD,2.0,'-');   % large stretch
cap(ax,'finite stretch,\ $\nabla\!\cdot\!P = 0$','corrupted: faulty boundary sensor',GY);

exportgraphics(fig,OUT,'Resolution',600,'BackgroundColor','white');
fprintf('saved %s\n',OUT);

%% ---- helpers ----
% panel: start a new tile as a clean, equal-aspect [0,1]^2 drawing canvas
function ax = panel()
    ax = nexttile; hold(ax,'on'); axis(ax,'equal'); xlim(ax,[0 1]); ylim(ax,[0 1]);
    axis(ax,'off'); set(ax,'Clipping','off');
end
% cap: write the governing equation (center) and the corruption note (below it)
function cap(ax,eq,corr,gy)
    text(ax,.5,.12,eq,'Interpreter','latex','FontSize',12,'HorizontalAlignment','center');
    text(ax,.5,.015,corr,'Interpreter','latex','FontSize',9.3,'Color',gy,'HorizontalAlignment','center');
end
% arrow: draw a line from (x0,y0) to (x1,y1) with a filled triangular head
function arrow(ax,x0,y0,x1,y1,col,lw,ls)
    line(ax,[x0 x1],[y0 y1],'Color',col,'LineWidth',lw,'LineStyle',ls);
    d=[x1-x0,y1-y0]; L=norm(d); u=d/L; p=[-u(2) u(1)]; hs=.032;   % unit direction u, perpendicular p, head size hs
    tip=[x1 y1]; b1=tip-hs*u*1.7+hs*p; b2=tip-hs*u*1.7-hs*p;   % arrowhead base corners
    patch(ax,[tip(1) b1(1) b2(1)],[tip(2) b1(2) b2(2)],col,'EdgeColor',col);
end
