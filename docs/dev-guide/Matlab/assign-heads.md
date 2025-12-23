
# assign_heads_to_worms

## What the function is trying to solve

You have:
- N worms (each worm has a skeleton/centerline polyline in measures{i,1})
- K clicked head points in head_coords (Kx2)

Goal:
- match at most one head to each worm and at most one worm to each head
- do it robustly when worms are close


## Step-by-step: what assign_heads_to_worms does

### 1) Read params and set defaults

If params is missing fields, it fills them with defaults:

- resample_step_px = 1.2
  - Resamples each worm's centerline to ~1.2 px spacing along arc length.

- max_dist_px = Inf
  - Optional hard cutoff on Euclidean distance from click -> worm skeleton.

- max_end_frac = 0.40
  - Only allow matches where the closest point is within 40% of the worm length from an end
    (near head/tail, not mid-body).

- lambda_end = 1.0
  - Adds a penalty for being far from an end (arc-length distance).


### 2) Normalize inputs

- H = ensure_xy(head_coords)
  - Ensures heads are Kx2 and removes NaN/Inf rows.
- measures{i,1} is similarly ensured as Nx2.


### 3) Preprocess each worm: resample and compute arc length

For each worm i:
- Take original centerline: C0 = measures{i,1}
- Resample it into Cres with roughly uniform spacing (interpolation along cumulative length)
- Compute:
  - cumlen: cumulative arc length (pixels) for each point in the resampled line
  - L: total worm length in pixels (cumlen(end))

Stored in:
- Worm(i).C      = Cres
- Worm(i).cumlen = cumlen
- Worm(i).L      = L


### 4) Build a full N x K matching table (cost + metadata)

Allocates matrices:

- D(i,j)    = final cost of assigning head j to worm i (starts as Inf)
- Dxy(i,j)  = Euclidean distance from head point to closest point on worm polyline
- Dend(i,j) = arc-length distance from that closest point to the nearest end
- Spos(i,j) = arc-length position (s_at) of the closest point (distance from start)
- Near(i,j,:) = the actual closest point coordinates on the skeleton

Then for every worm i and head j:

1) Compute the closest point on the polyline (exact segment projection):
   - [dxy, s_at, p_near] = point_to_polyline(pt, C, cum)

2) Compute "distance to nearest end" along the skeleton:
   - dend = min(s_at, L - s_at)

3) Gate (disallow) bad matches:
   - must satisfy: dxy <= max_dist_px
   - and dend/L <= max_end_frac
     (closest point must be near an end)

4) If allowed, compute the cost:
   - cost = dxy + lambda_end * dend

So it prefers:
- clicks close to the skeleton (small dxy)
- and close to an end (small dend)


### 5) Solve the global assignment (one-to-one matching)

Now it has a cost matrix D (Inf = forbidden).

- If MATLAB's matchpairs exists:
  - use it to find the minimum total cost matching
- Otherwise:
  - fall back to a greedy matcher ("pick smallest available cost, then delete that row/col")

Result:
- pairs is Mx2 rows of [worm_idx, head_idx]


### 6) Materialize outputs

For each matched pair (i,j):
- heads_by_worm(i,:) = H(j,:)

Decide whether that head is closer to the start end or end end of the worm:
- is_start_by_worm(i) = (s_at <= L - s_at)
  (if the nearest point lies closer to the start half, it marks start)

Then it computes:
- unmatched_worms: worms with NaN head assigned
- unmatched_heads: heads not used


## What diag means

Diag is a diagnostics struct returned so you can inspect why matches happened (or didn't).

It's created by pack_diag(...):

- diag = struct(
    'cost', D,
    'd_euclid', Dxy,
    'd_to_end_px', Dend,
    's_at_px', Spos,
    'nearest_xy', Near,
    'worm_lengths_px', [Worm.L].'
  )

So:

- diag.cost (N x K)
  - final cost used for matching (dxy + lambda_end*dend), Inf if gated out

- diag.d_euclid (N x K)
  - Euclidean distance from head click to closest point on worm skeleton

- diag.d_to_end_px (N x K)
  - arc-length distance from that closest point to the nearest end

- diag.s_at_px (N x K)
  - arc-length position along the skeleton (0 at start, L at end) of the closest point

- diag.nearest_xy (N x K x 2)
  - the closest point coordinates on the resampled skeleton

- diag.worm_lengths_px (N x 1)
  - each worm's total skeleton length in pixels


## How you'd use diag in practice

If a head got assigned to the "wrong" worm, for a given head index j you can compare:
- which worm had the smallest diag.d_euclid(:,j)
- whether the "correct" worm got gated out because:
  diag.d_to_end_px(:,j) / diag.worm_lengths_px > max_end_frac
- whether lambda_end is dominating the decision (large dend penalty)

