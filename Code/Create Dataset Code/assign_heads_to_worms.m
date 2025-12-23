function [heads_by_worm, is_start_by_worm, pairs, unmatched_worms, unmatched_heads, diag] = ...
    assign_heads_to_worms(measures, head_coords, params)
% Robustly assign heads (Kx2) to worms (measures Nx2 cell: {centerline Nx2, width_pts 2x2})
% using nearest point on a finely-interpolated skeleton + end proximity bias.
%
% params (all optional):
%   .resample_step_px  (default 1.2)   % arc spacing for centerline resample
%   .max_dist_px       (default Inf)   % gate by absolute distance (pixels)
%   .max_end_frac      (default 0.40)  % gate by arc fraction to nearest end
%   .lambda_end        (default 1.0)   % cost weight for arc distance to end (px)
%
% Outputs:
%   heads_by_worm    : N×2 [x y] (NaN if unmatched)
%   is_start_by_worm : N×1 logical; true if closer to start end
%   pairs            : M×2 [worm_idx, head_idx]
%   unmatched_worms  : vector of worm indices with no head
%   unmatched_heads  : vector of head indices unused
%   diag             : struct with per-pair diagnostics (optional)

% ---- defaults ----
if nargin < 3, params = struct; end
P = defaults(params, struct( ...
    'resample_step_px', 1.2, ...
    'max_dist_px',      inf, ...
    'max_end_frac',     0.40, ...
    'lambda_end',       1.0 ...
));

N = size(measures,1);
H = ensure_xy(head_coords);
K = size(H,1);

heads_by_worm    = nan(N,2);
is_start_by_worm = false(N,1);
pairs = zeros(0,2);
unmatched_worms = (1:N).';
unmatched_heads = (1:K).';

if N==0 || K==0
    diag = struct();
    return
end

% ---- preprocess worms: resample & cumulative arc length ----
Worm(N) = struct('C',[],'cumlen',[],'L',0); %#ok<NASGU>
for i = 1:N
    C0 = ensure_xy(measures{i,1});
    [Cres, ~, cumlen] = resample_centerline_with_cumlen(C0, P.resample_step_px);
    Worm(i).C      = Cres;
    Worm(i).cumlen = cumlen;
    Worm(i).L      = cumlen(end);
end

% ---- build N×K cost and metadata ----
D   = inf(N,K);    % final cost
Dxy = inf(N,K);    % euclidean distance to polyline (px)
Dend= inf(N,K);    % arc distance to nearest end (px)
Spos= nan(N,K);    % arc position s_at (px from start)
Near = nan(N,K,2); % nearest point coords

for i = 1:N
    C   = Worm(i).C;
    cum = Worm(i).cumlen;
    L   = Worm(i).L;
    for j = 1:K
        pt = H(j,:);
        [dxy, s_at, p_near] = point_to_polyline(pt, C, cum); % exact segment projection
        % arc distance to nearest end (px)
        dend = min(s_at, L - s_at);
        % gating
        if dxy <= P.max_dist_px && (dend / max(L,eps)) <= P.max_end_frac
            cost = dxy + P.lambda_end * dend;
            D(i,j)    = cost;
            Dxy(i,j)  = dxy;
            Dend(i,j) = dend;
            Spos(i,j) = s_at;
            Near(i,j,:) = p_near;
        end
    end
end

% ---- global assignment ----
if exist('matchpairs','file') == 2
    % matchpairs minimizes cost; we must ignore Inf entries.
    finite_mask = isfinite(D);
    if ~any(finite_mask,'all')
        diag = pack_diag(Worm, D, Dxy, Dend, Spos, Near);
        return
    end
    % Use a huge threshold to let costs decide while respecting Inf as disallowed
    maxCost = max(D(finite_mask));
    [mpairs, ~] = matchpairs(D, maxCost + 1);  % returns [worm_idx, head_idx]
    pairs = mpairs;
else
    pairs = greedy_assign(D);
end

% ---- materialize assignments ----
used_heads = false(1,K);
for r = 1:size(pairs,1)
    i = pairs(r,1); j = pairs(r,2);
    heads_by_worm(i,:) = H(j,:);
    used_heads(j) = true;
    % orientation from arc position
    s_at = Spos(i,j);
    L    = Worm(i).L;
    is_start_by_worm(i) = (s_at <= L - s_at);
end

unmatched_worms = find(isnan(heads_by_worm(:,1)));
unmatched_heads = find(~used_heads);

diag = pack_diag(Worm, D, Dxy, Dend, Spos, Near);

end

% ================= helpers ==================

function P = defaults(P, D)
    f = fieldnames(D);
    for k = 1:numel(f)
        if ~isfield(P,f{k}) || isempty(P.(f{k})), P.(f{k}) = D.(f{k}); end
    end
end

function A = ensure_xy(A)
    A = double(A);
    if isempty(A), A = zeros(0,2); return, end
    if size(A,2)==2
        % ok
    elseif size(A,1)==2
        A = A.';
    else
        error('Expected [N×2] or [2×N] coordinates.');
    end
    A = A(all(isfinite(A),2),:);
end

function [C, s_norm, cumlen] = resample_centerline_with_cumlen(C0, step)
    % uniform-ish spacing, cumlen in px, s_norm in [0,1]
    if size(C0,1) < 2
        C = C0; cumlen = 0; s_norm = 0; return
    end
    d  = hypot(diff(C0(:,1)), diff(C0(:,2)));
    L  = [0; cumsum(d)];
    tq = (0:step:L(end))'; if tq(end) < L(end), tq(end+1)=L(end); end
    C  = [interp1(L, C0(:,1), tq, 'linear'), interp1(L, C0(:,2), tq, 'linear')];
    cumlen = tq;
    s_norm = tq / max(tq(end), eps);
    % ensure unique rows
    C = unique(C,'rows','stable');
    % if too short, pad a hair to avoid degeneracy
    if size(C,1) < 2
        C = [C; C + [1e-6,0]];
        cumlen = [0; step];
        s_norm = cumlen / step;
    end
end

function [dmin, s_at, p_near] = point_to_polyline(pt, C, cumlen)
% Exact shortest distance from point to polyline via segment projection.
% Returns:
%   dmin : distance (px)
%   s_at : arc position (px from start) of the closest point
%   p_near: [x y] coordinates of the closest point
    M = size(C,1);
    dmin2 = inf; s_at = 0; p_near = [NaN NaN];
    for k = 1:M-1
        a = C(k,:); b = C(k+1,:);
        ab = b - a;
        ab2 = ab(1)^2 + ab(2)^2;
        if ab2 < eps
            % degenerate tiny segment
            t = 0; proj = a;
        else
            t = ((pt - a) * ab.') / ab2;       % scalar projection
            t = min(max(t,0),1);               % clamp to [0,1]
            proj = a + t * ab;
        end
        d2 = (pt(1)-proj(1))^2 + (pt(2)-proj(2))^2;
        if d2 < dmin2
            dmin2 = d2;
            p_near = proj;
            s_at   = cumlen(k) + t * (cumlen(k+1) - cumlen(k));
        end
    end
    dmin = sqrt(dmin2);
end

function pairs = greedy_assign(D)
% Simple greedy assignment on finite costs
    [N,K] = size(D);
    pairs = zeros(0,2);
    Dwork = D;
    while true
        [v, idx] = min(Dwork(:));
        if ~isfinite(v), break; end
        [i, j] = ind2sub([N,K], idx);
        pairs(end+1,:) = [i, j]; %#ok<AGROW>
        Dwork(i,:) = inf;
        Dwork(:,j) = inf;
    end
end

function diag = pack_diag(Worm, D, Dxy, Dend, Spos, Near)
    diag = struct('cost', D, 'd_euclid', Dxy, 'd_to_end_px', Dend, ...
                  's_at_px', Spos, 'nearest_xy', Near, ...
                  'worm_lengths_px', [Worm.L].');
end
