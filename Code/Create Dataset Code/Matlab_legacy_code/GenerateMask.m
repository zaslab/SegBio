function [worm_mask, head_mask, tail_mask, neck_mask, head_boundary,...
    tail_boundary, bbox, C, out] = GenerateMask(im, measures_row, head_coords, params)
% GenerateMask — pure, headless, asymmetric width support.
%
% Inputs
%   im            : H×W or H×W×3 (only size is used)
%   measures_row  : 1×2 cell {centerline Nx2, width_pts 2×2} OR struct with
%                   .centerline (Nx2), .width_pts (2×2) [, .width_profile (optional)]
%   head_coords   : [x y] OR K×2 list of candidate heads for the whole image
%   params        : struct (all optional)
%       .resample_step_px     (1.5)
%       .morph_close_radius   (2)
%       .hole_fill            (true)
%       .head_frac            (0.18)   % arc fraction for head mask length
%       .neck_halfwidth_frac  (0.03)   % arc half-width around head boundary
%       .end_min_frac         (0.12)   % minimum tip diameter (× maxW)
%       .taper_head           (1.8)    % sharper → larger
%       .taper_tail           (1.2)    % slower → smaller
%       .taper_shape          ('poly') % 'poly' or 'exp'
%       .max_head_dist        (inf)    % gate in pixels for pairing to a head
%       .mean_worm            (optional struct with fields):
%                               .to_head [M1×1] relative diam from peak to head
%                               .to_tail [M2×1] relative diam from peak to tail
%
% Outputs
%   worm_mask, head_mask, tail_mask, neck_mask : H×W logical
%   bbox       : [xmin ymin width height]
%   C          : resampled & oriented centerline (M×2), C(1,:) is head
%   out        : struct with diagnostics:
%                  .idx_peak, .s_peak, .maxW, .picked_head, .is_start,
%                  .paired_head_idx (if head_coords list given)
%
% Notes
% - No UI; will error if no head can be paired within gate (if gate finite).
% - If params.mean_worm is provided, that profile overrides polynomial/exp taper.

% ---------------- defaults ----------------
if nargin < 4, params = struct; end
P = defaults(params, struct( ...
    'resample_step_px',    1.5, ...
    'morph_close_radius',  2, ...
    'hole_fill',           true, ...
    'head_frac',           0.18, ...
    'neck_halfwidth_frac', 0.03, ...
    'end_min_frac',        0.12, ...
    'taper_head',          1.8, ...
    'taper_tail',          1.2, ...
    'taper_shape',         'poly', ...
    'max_head_dist',       inf ...
));

% ------------- parse measures row ----------
[C0, maxW, w_profile_opt, width_pts] = parse_measures_row(measures_row);
H = size(im,1); W = size(im,2);

% ------------- resample centerline ---------
[C, s] = resample_centerline(C0, P.resample_step_px); % s in [0,1]
if size(C,1) < 2
    error('Centerline has <2 points after resampling.');
end

% ------------- pick head for THIS worm -----

[picked_head, is_start, head_idx] = pick_head_for_worm(C, head_coords, P.max_head_dist);
out.picked_head       = picked_head;
out.is_start          = is_start;
out.paired_head_idx   = head_idx;

% orient so index 1 is the head
if ~is_start
    C  = flipud(C);
    s  = flipud(1 - s);
end

% ------------- locate widest point (peak) --
mid = mean(width_pts(1:2,:),1); % midpoint of the width clicks
[~, idx_peak] = min(hypot(C(:,1)-mid(1), C(:,2)-mid(2)));
idx_peak = ceil(length(s)/2);
s_peak = s(idx_peak);
out.idx_peak = idx_peak;
out.s_peak   = s_peak;
out.maxW     = maxW;

% ------------- build asymmetric width ------
if ~isempty(w_profile_opt)
    % If a per-point profile was provided in measures, resample it.
    w_rel = resample_profile(w_profile_opt(:), numel(s));
    % Ensure the peak is at 1 (relative), normalize 
    m = max(w_rel); if m>0, w_rel = w_rel/m; end
    % Optional: shift peak to idx_peak if needed (rare); here we trust provided profile.
    w_px = max(1, maxW * max(P.end_min_frac, w_rel));
elseif isfield(P,'mean_worm') && ~isempty(P.mean_worm)
    w_px = profile_from_template(s, idx_peak, maxW, P.mean_worm, P.end_min_frac);
else
    w_px = asymmetric_width_profile(s, idx_peak, maxW, ...
               P.end_min_frac, P.taper_head, P.taper_tail, P.taper_shape);
end
w_px = w_px;
% ------------- full body polygon → mask ----
worm_poly = centerline_to_polygon(C, w_px);
worm_mask = poly2mask(worm_poly(:,1), worm_poly(:,2), H, W);
worm_mask = postprocess(worm_mask, P.morph_close_radius, P.hole_fill);

% ------------- head / neck / tail ----------
M = size(C,1);
idx_head_end = max(3, min(M, round(P.head_frac * (M-1))));
neck_half    = max(2, round(P.neck_halfwidth_frac * (M-1)));
idx_neck_lo  = max(2, idx_head_end - neck_half);
idx_neck_hi  = min(M-1, idx_head_end + neck_half);

head_poly = centerline_to_polygon(C(1:idx_head_end,:), w_px(1:idx_head_end));
head_mask = poly2mask(head_poly(:,1), head_poly(:,2), H, W);
head_mask = head_mask & worm_mask;
head_boundary = GetEndBoundary(head_mask);

neck_poly = centerline_to_polygon(C(idx_neck_lo:idx_neck_hi,:), w_px(idx_neck_lo:idx_neck_hi));
neck_mask = poly2mask(neck_poly(:,1), neck_poly(:,2), H, W);
neck_mask = neck_mask & worm_mask;

idx_tail_neck = (M-idx_head_end - neck_half):(M-idx_head_end + neck_half);
tail_neck_poly = centerline_to_polygon(C(idx_tail_neck,:), w_px(idx_tail_neck));
neck_tail_mask = poly2mask(tail_neck_poly(:,1), tail_neck_poly(:,2), H, W);
neck_mask = neck_mask | neck_tail_mask;


tail_poly = centerline_to_polygon(C((M-idx_head_end):end,:), w_px((M-idx_head_end):end));
tail_mask = poly2mask(tail_poly(:,1), tail_poly(:,2), H, W);
tail_mask  = tail_mask  & worm_mask;
tail_boundary = GetEndBoundary(tail_mask);
% ------------- bbox ------------------------
stats = regionprops(worm_mask,'BoundingBox');
bbox = [0 0 0 0]; if ~isempty(stats), bbox = stats(1).BoundingBox; end
end

% =================== helpers ===================
function bound = GetEndBoundary(mask)
% 1) 1-pixel-wide outline on the *inside* of the object
inner = bwperim(mask, 8);          % logical mask
% 2) 1-pixel-wide outline just *outside* the object
se    = strel('square', 3);      % 8-connected neighborhood
outer = imdilate(mask, se) & ~mask;  % dilated object minus original
% 3) Combined 2-pixel boundary layer (inner+outer)
bound = inner | outer;
end

function P = defaults(P, D)
    f = fieldnames(D);
    for i=1:numel(f)
        k = f{i};
        if ~isfield(P,k) || isempty(P.(k)), P.(k) = D.(k); end
    end
end

function [C0, maxW, w_profile, width_pts] = parse_measures_row(mr)
    w_profile = []; width_pts = [];
    if iscell(mr)
        assert(numel(mr)>=2, 'measures_row must be a 1×2 cell: {centerline, width_pts}.');
        C0 = ensure_xy(mr{1});
        width_pts = ensure_xy(mr{2});
    elseif isstruct(mr)
        assert(isfield(mr,'centerline') && isfield(mr,'width_pts'), ...
            'struct must have .centerline (Nx2) and .width_pts (2x2).');
        C0 = ensure_xy(mr.centerline);
        width_pts = ensure_xy(mr.width_pts);
        if isfield(mr,'width_profile'), w_profile = mr.width_profile; end
    else
        error('measures_row must be cell or struct.');
    end
    assert(size(width_pts,1)>=2, 'width_pts must contain two points (2×2).');
    width_pts = width_pts(1:2,:);
    maxW = hypot(width_pts(2,1)-width_pts(1,1), width_pts(2,2)-width_pts(1,2));
end

function A = ensure_xy(A)
    A = double(A);
    if size(A,2)==2
        % ok
    elseif size(A,1)==2
        A = A.';
    else
        error('Expected Nx2 or 2×N [x y] array.');
    end
    A = A(all(isfinite(A),2),:);
end

function [C, s] = resample_centerline(C0, step)
    d  = hypot(diff(C0(:,1)), diff(C0(:,2)));
    L  = [0; cumsum(d)];
    if L(end) < eps
        C = C0; s = linspace(0,1,size(C,1))'; return
    end
    tq = (0:step:L(end))';
    if tq(end) < L(end), tq(end+1) = L(end); end
    C  = [interp1(L, C0(:,1), tq, 'linear'), ...
          interp1(L, C0(:,2), tq, 'linear')];
    s  = tq / L(end);
    % ensure ≥3 pts for robust normals
    C = unique(C,'rows','stable');
    if size(C,1) < 3, C = [C; C(end,:)+[1e-6,0]]; end
end

function [picked_head, is_start, head_idx] = pick_head_for_worm(C, head_coords, max_dist)
% Accept either a single [x y] or K×2 list; pick nearest to endpoints.
    head_idx = NaN;
    if isempty(head_coords)
        error('head_coords is empty — provide a head point or a list of heads for this image.');
    end
    H = double(head_coords);
    if size(H,2) ~= 2, error('head_coords must be [x y] or K×2.'); end
    if size(H,1) == 1
        picked_head = H(1,:);
        d1 = hypot(picked_head(1)-C(1,1),  picked_head(2)-C(1,2));
        d2 = hypot(picked_head(1)-C(end,1),picked_head(2)-C(end,2));
        is_start = (d1 <= d2);
        if d1>max_dist && d2>max_dist
            error('Provided head is farther than max_head_dist from both endpoints.');
        end
        return
    end
    % K×2 case: compute distance to endpoints and choose nearest
    dstart = hypot(H(:,1)-C(1,1),  H(:,2)-C(1,2));  % K×1
    dend   = hypot(H(:,1)-C(end,1),H(:,2)-C(end,2));
    dmin   = min(dstart, dend);
    [d_best, j] = min(dmin);
    if d_best > max_dist
        error('No head within max_head_dist for this worm.');
    end
    picked_head = H(j,:);
    head_idx    = j;
    is_start    = (dstart(j) <= dend(j));
end

function w_px = asymmetric_width_profile(s, idx_peak, maxW, end_min_frac, taper_head, taper_tail, shape)
% Build relative width piecewise around idx_peak with independent tapers.
    M = numel(s);
    w_rel = zeros(M,1);
    % head side (indices 1..idx_peak)
    if idx_peak > 1
        u_head = (s(1:idx_peak) - s(1)) / max(s(idx_peak) - s(1), 1e-6); % 0..1 head→peak
        w_rel(1:idx_peak) = 1 - falloff(u_head, taper_head, shape);
    else
        w_rel(1) = 1;
    end
    % tail side (idx_peak..M)
    if idx_peak < M
        u_tail = (s(idx_peak:end) - s(idx_peak)) / max(s(end) - s(idx_peak), 1e-6); % 0..1 peak→tail
        w_rel(idx_peak:end) = 1 - falloff(u_tail, taper_tail, shape);
    end
    w_rel(idx_peak) = 1; % enforce exact peak
    w_rel = max(end_min_frac, w_rel);
    w_px  = max(1, maxW * w_rel(:));
end

function y = falloff(u, p, shape)
    u = min(max(u,0),1);
    switch lower(shape)
        case 'poly'
            y = u.^p;                   % 0→0, 1→1, steeper with larger p
        case 'exp'
            k = max(0.01, p);
            y = (exp(k*u)-1) / (exp(k)-1);
        otherwise
            y = u.^p;
    end
end

function w_px = profile_from_template(s, idx_peak, maxW, tmpl, end_min_frac)
% Use a supplied mean_worm template: to_head & to_tail relative diameters.
% Resample each side to match the number of samples on that side.
    M = numel(s);
    w_rel = zeros(M,1);
    % Head side
    nH = max(1, idx_peak);
    if isfield(tmpl,'to_head') && ~isempty(tmpl.to_head)
        vecH = tmpl.to_head(:);
        w_rel(1:nH) = resample_profile(vecH, nH);
    else
        w_rel(1:nH) = linspace(end_min_frac, 1, nH);
    end
    % Tail side
    nT = max(1, M - idx_peak + 1);
    if isfield(tmpl,'to_tail') && ~isempty(tmpl.to_tail)
        vecT = tmpl.to_tail(:);
        w_rel(idx_peak:M) = resample_profile(vecT, nT);
    else
        w_rel(idx_peak:M) = linspace(1, end_min_frac, nT);
    end
    % Ensure peak is 1 and enforce min
    w_rel(idx_peak) = 1;
    w_rel = max(end_min_frac, w_rel);
    % Normalize (template may not be exactly 1)
    m = max(w_rel); if m>0, w_rel = w_rel/m; end
    w_px = max(1, maxW * w_rel(:));
end

function v = resample_profile(v_in, N)
    if numel(v_in) == N
        v = v_in(:);
    else
        x = linspace(0,1,numel(v_in));
        v = interp1(x, v_in, linspace(0,1,N), 'linear','extrap').';
    end
end

function poly = centerline_to_polygon(C, w_px)
% Offset by ± normal*(w/2) and build a closed polygon.
    M = size(C,1); w_px = w_px(:);
    if numel(w_px) ~= M, w_px = w_px(1)*ones(M,1); end
    T = zeros(M,2);
    T(2:M-1,:) = C(3:M,:) - C(1:M-2,:);
    T(1,:)     = C(2,:)   - C(1,:);
    T(M,:)     = C(M,:)   - C(M-1,:);
    nrm = hypot(T(:,1), T(:,2)) + eps;
    T = T ./ nrm;
    N = [-T(:,2), T(:,1)];
    halfw = 0.5 * w_px;
    left  = C + N .* halfw;
    right = C - N .* halfw;
    poly = [left; flipud(right)];
    poly = poly(all(isfinite(poly),2), :);
end

function BW = postprocess(BW, close_r, do_fill)
% Single morphology pass; persistent strel cache.
    if close_r > 0
        persistent se_cache r_cache
        if isempty(se_cache) || r_cache~=close_r
            se_cache = strel('disk', close_r, 0);
            r_cache  = close_r;
        end
        BW = imclose(BW, se_cache);
    end
    if do_fill, BW = imfill(BW, 'holes'); end
    BW = BW>0;
end

