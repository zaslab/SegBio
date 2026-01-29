function save_training_images(indir, outdir, opts)
% save_training_images  Headless, parallel-friendly exporter for NN data
%
% Inputs
%   indir   : folder to scan (recursively) for *measures*.mat files
%   outdir  : destination root folder
%   opts    : struct (all optional)
%       .interactive     (true)   % click heads only if missing
%       .overwrite       (false)
%       .use_parfor      (auto)   % use parfor if Parallel Toolbox available
%       .max_head_dist   (80)     % px gate for head→skeleton match
%       .max_end_frac    (0.40)   % ≤ this fraction of length from an end
%       .lambda_end      (1.0)    % cost weight for arc-dist-to-end (px)
%       .params          (struct) % passed into GenerateMask (taper, etc.)
%       .strain          ('unknown') % saved in strain.mat
%       .manifest_name   ('manifest.csv') % optional index written to outdir
%
% Output layout (one UUID folder per image under outdir):
%   in.mat           -> variable 'im'
%   out.mat          -> variable 'out'           (uint16 instance labels 1..N)
%   heads_mask.mat   -> variable 'heads_mask'    (logical)
%   tails_mask.mat   -> variable 'tails_mask'    (logical)
%   ignore_mask.mat  -> variable 'ignore_mask'   (logical)
%   box.mat          -> variable 'box'           (Nx4 [xmin ymin width height])
%   head_coords.mat  -> variable 'head_coords'   (Kx2)
%   strain.mat       -> variable 'strain'        (string/char)
%
% Requires on path: GenerateMask.m, assign_heads_to_worms.m
% Optional (only if opts.interactive=true): MarkHeadsInteractive.m

if nargin < 3, opts = struct; end
opts = def(opts, 'interactive',  true);
opts = def(opts, 'overwrite',    false);
opts = def(opts, 'use_parfor',   parallel_available());
opts = def(opts, 'max_head_dist', 80);
opts = def(opts, 'max_end_frac',  0.40);
opts = def(opts, 'lambda_end',    1.0);
opts = def(opts, 'params',        struct);
opts = def(opts, 'strain',        'unknown');
opts = def(opts, 'manifest_name', 'manifest.csv');

% Create .csv to track process
if ~exist(outdir,'dir'), mkdir(outdir); end
manifest_path = fullfile(outdir, opts.manifest_name);
append_manifest_header_if_needed(manifest_path);

meas = dir(fullfile(indir, '**', '*measures*.mat'));
if isempty(meas)
    warning('No *measures*.mat files found under: %s', indir);
    return
end

skipped = {};

for i = 1:numel(meas)
    mpath = fullfile(meas(i).folder, meas(i).name);
    strain_folder = meas(i).folder;
    strain_ind = strfind(meas(i).folder, '\');
    opts.strain = char(strain_folder(strain_ind(end)+1:end));
    % --- find the matching image file ---
    [im_path, sample_id] = find_image_for_measures(mpath);
    if isempty(im_path)
        warning('No image found for measures: %s', mpath);
        skipped{end+1} = mpath; 
        continue
    end

    % --- load measures ---
    S = load(mpath);
    if ~isfield(S,'measures')
        warning('Missing variable ''measures'' in %s', mpath);
        skipped{end+1} = mpath; 
        continue
    end
    measures = S.measures;
    if isempty(measures) || size(measures,2) < 2
        warning('Bad/empty measures in %s', mpath);
        skipped{end+1} = mpath; 
        continue
    end

    % --- load image ---
    im = imread(im_path);
    [H,W,~] = size(im);

    % --- get head coords (from sibling file or interactive if allowed) ---
    hc_path = fullfile(fileparts(mpath), [sample_id 'head_coords.mat']);
    head_coords = load_if(hc_path, 'head_coords', []);
    if isempty(head_coords)
        if opts.interactive
            assert(exist('MarkHeadsInteractive','file')==2, ...
                'opts.interactive=true but MarkHeadsInteractive.m is not on the path.');
            [head_coords, canceled] = MarkHeadsInteractive(im, measures, [], ...
                struct('expected_count', size(measures,1), ...
                       'save_path', hc_path, ...
                       'title', sprintf('Mark heads: %s', sample_id)));
            if canceled
                fprintf('Canceled: %s\n', sample_id);
                continue
            end
        else
            fprintf('No head_coords; skipping (interactive=false): %s\n', sample_id);
            skipped{end+1} = mpath; 
            continue
        end
    end

    % --- robust head→worm assignment ---
    [heads_by_worm, is_start_by_worm, pairs, um_w, um_h] = assign_heads_to_worms( ...
        measures, head_coords, struct('max_dist_px', opts.max_head_dist, ...
                                      'max_end_frac', opts.max_end_frac, ...
                                      'lambda_end',   opts.lambda_end)); %#ok<ASGLU>

    if ~isempty(um_w)
        fprintf('Warning: %d worms unmatched in %s (skipping unmatched)\n', numel(um_w), sample_id);
        logpath = indir+ "\Missed.csv";
        row = {mpath};  % <-- your values in column order
        Trow = cell2table(row, 'VariableNames', "Path");
        writetable(Trow, logpath, 'WriteMode','append', 'WriteVariableNames', false);
    end

    N = size(measures,1);
    % per-worm outputs as cells for parfor safety
    WM  = cell(N,1); HM = cell(N,1); TM = cell(N,1); NK = cell(N,1);
    HB = cell(N,1); TB = cell(N,1);
    BOX = nan(N,4);

    % --- generate masks per worm (parfor optional) ---
    if opts.use_parfor
        parfor w = 1:N
            [WM{w}, HM{w}, TM{w}, NK{w}, HB{w}, TB{w}, BOX(w,:)] = gen_one_worm(w, im, measures, heads_by_worm, opts.params, sample_id);
        end
    else
        for w = 1:N
            [WM{w}, HM{w}, TM{w}, NK{w}, HB{w}, TB{w}, BOX(w,:)] = gen_one_worm(w, im, measures, heads_by_worm, opts.params, sample_id);
        end
    end

    % --- aggregate instance map and class masks ---
    out_inst    = zeros(H,W,'uint16');   % 1..N instance labels
    heads_mask  = false(H,W);
    tails_mask  = false(H,W);
    ignore_mask = false(H,W);
    head_bound = false(H,W);
    tail_bound = false(H,W);

    for w = 1:N
        if isempty(WM{w}), continue; end
        out_inst    = max(out_inst, uint16(w) .* uint16(WM{w}));
        heads_mask  = heads_mask  | HM{w};
        tails_mask  = tails_mask  | TM{w};
        ignore_mask = ignore_mask | NK{w};
        head_bound = head_bound | HB{w};
        tail_bound = tail_bound | TB{w};
        
    end

    % --- choose a fresh UUID folder and save the MAT bundle there ---
    uuid = char(java.util.UUID.randomUUID);
    sample_dir = fullfile(outdir, uuid);
    if ~exist(sample_dir,'dir'), mkdir(sample_dir); end

    % If not overwriting, skip when out.mat already exists
    if ~opts.overwrite && exist(fullfile(sample_dir,'out.mat'),'file')
        fprintf('Exists, skipping: %s\n', uuid);
        continue
    end

    save_mat_bundle(sample_dir, im, out_inst, heads_mask, tails_mask,...
        ignore_mask, head_bound, tail_bound, BOX, head_coords, opts.strain);

    % --- append to manifest (uuid identifies this sample) ---
    has_mask = ~cellfun(@isempty, WM);
    append_manifest_row(manifest_path, uuid, im_path, sample_dir, sum(has_mask));
end

% optional log of skipped items
if ~isempty(skipped)
    fid = fopen(fullfile(outdir, 'skipped.txt'),'a');
    for k = 1:numel(skipped), fprintf(fid, "%s\n", skipped{k}); end
    fclose(fid);
end
end

% ======================= nested / local helpers =======================

function [wm, hm, tm, nk, hb, tb, box] = gen_one_worm(w, im, measures, heads_by_worm, gen_params, sample_id)
% wrapper around GenerateMask for one worm
wm=[]; hm=[]; tm=[]; nk=[]; hb=[]; tb=[]; box=[nan nan nan nan];
hc = heads_by_worm(w,:);
if any(isnan(hc)), return; end
try
    [wm, hm, tm, nk, hb, tb, box] = GenerateMask(im, measures(w,:), hc, gen_params);
catch ME
    warning('GenerateMask failed for worm %d (%s): %s', w, sample_id, ME.message);
end
end

function S = def(S, field, default)
% Set default field value into struct if missing/empty.
    if ~isfield(S, field) || isempty(S.(field))
        S.(field) = default;
    end
end

function tf = parallel_available()
tf = ~isempty(ver('parallel')) || ~isempty(ver('distcomp'));
end

function val = load_if(path, varname, default)
if exist(path,'file')
    try
        tmp = load(path, varname);
        if isfield(tmp, varname), val = tmp.(varname); else, val = default; end
    catch
        val = default;
    end
else
    val = default;
end
end

function [img_path, sample_id] = find_image_for_measures(mpath)
% Heuristic resolver: remove "measures" token from filename to get base,
% look for first image with that base in the same folder; else pick single image.
[folder, name, ~] = fileparts(mpath);
base = regexprep(name, 'measures', '', 'ignorecase');
base = regexprep(base, '_+$', '');
exts = {'tif','tiff','png','jpg','jpeg','bmp'};
img_path = '';
for e = exts
    c = dir(fullfile(folder, [base '*.' e{1}]));
    if ~isempty(c)
        img_path = fullfile(c(1).folder, c(1).name);
        break
    end
end
if isempty(img_path)
    c = dir(fullfile(folder, '*.*'));
    c = c(~[c.isdir]);
    keep = false(size(c));
    for k = 1:numel(c)
        [~,~,ex] = fileparts(c(k).name);
        keep(k) = any(strcmpi(ex(2:end), exts));
    end
    c = c(keep);
    if numel(c)==1
        img_path = fullfile(c(1).folder, c(1).name);
    end
end
if ~isempty(img_path)
    [~, nm, ~] = fileparts(img_path);
    sample_id = nm;
else
    sample_id = name;
end
end

function append_manifest_header_if_needed(path)
hdr = 'uuid,image_path,sample_dir,num_worms';
if ~exist(path,'file')
    fid = fopen(path,'w'); fprintf(fid, "%s\n", hdr); fclose(fid);
end
end

function append_manifest_row(path, uuid, im_path, sample_dir, n_worms)
fid = fopen(path,'a');
fprintf(fid, "%s,%s,%s,%d\n", uuid, escape_csv(im_path), escape_csv(sample_dir), n_worms);
fclose(fid);
end

function s = escape_csv(x)
s = string(x);
s = replace(s, "\", "/");         % portability
s = replace(s, """", """""");     % escape quotes for CSV
end

function save_mat_bundle(sample_dir, in, out_inst, heads_mask, tails_mask,...
    ignore_mask, head_bound, tail_bound, BOX, head_coords, strain_val)
% Writes exactly the requested files to sample_dir as MAT variables.

% image
save(fullfile(sample_dir, 'in.mat'), 'in');

% instance map
out = out_inst; 
save(fullfile(sample_dir, 'out.mat'), 'out');

% class masks
save(fullfile(sample_dir, 'heads_mask.mat'),   'heads_mask');
save(fullfile(sample_dir, 'tails_mask.mat'),   'tails_mask');
save(fullfile(sample_dir, 'ignore_mask.mat'),  'ignore_mask');
save(fullfile(sample_dir, 'head_bound.mat'),  'head_bound')
save(fullfile(sample_dir, 'tail_bound.mat'),  'tail_bound')

% boxes
box = BOX; 
save(fullfile(sample_dir, 'box.mat'), 'box');

% heads
if nargin < 7 || isempty(head_coords)
    head_coords = zeros(0,2); 
end
save(fullfile(sample_dir, 'head_coords.mat'), 'head_coords');

% strain
if nargin < 8 || isempty(strain_val), strain_val = 'unknown'; end
strain = char(strain_val); 
save(fullfile(sample_dir, 'strain.mat'), 'strain');
end

