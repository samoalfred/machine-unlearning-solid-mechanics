function setup
%SETUP  Add the code folders to the MATLAB path. Run once per session from the
%   repository root, before calling any script by name. run_all calls it for you.
%   The legacy/ folder (superseded pre-revision scripts) is not added.
root = fileparts(mfilename('fullpath'));
for d = {'main','analysis','appendix','figures','template'}
    addpath(fullfile(root, d{1}));
end
if ~exist(fullfile(root,'results'),'dir'), mkdir(fullfile(root,'results')); end
cd(root);   % scripts read and write results/ relative to the repository root
fprintf('Code folders added to the path; working folder is %s\n', root);
end
