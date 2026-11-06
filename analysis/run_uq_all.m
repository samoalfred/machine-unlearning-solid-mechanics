%% Driver: rebuild every UQ ensemble (CS1-6) and the appendix UQ table/figure.
%% Runs uq_all_ff.m for CS1-4, uq_all_pinn.m for CS5-6, then uq_summary.m.
%% Each ensemble now uses one fixed test set shared by all members.
clear; clc;
for c = 1:4
    UQ_CASE = c; uq_all_ff;      %#ok<NASGU>
end
clear UQ_CASE
for c = 1:2
    UQP_CASE = c; uq_all_pinn;   %#ok<NASGU>
end
uq_summary;
