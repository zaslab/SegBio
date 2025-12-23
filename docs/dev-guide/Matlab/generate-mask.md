# Generate mask

The basic idea is to take the manually labeled center line and width and extrapolate from it to the entire worm.

The center line is interpolated to many points and the mask is grown perpendicularly out of the line in both directions for half the width.

Widest point (middle of width mark) is used as the max width (=1) and from there the mask tapers to both ends by fractions of the max, based on the mean_worm variable.

The tapering is asymmetrical and its direction is determined by the head markings in head_coords.

The tapering was manually determined and can be changed if needed.

*Output note: The head/tail/neck masks were used in the Mask R-CNN versions of the segmentor as additional input. Removing the neck pixels from the cost function gave the network some flexibility to decide where the head\tail starts.


Here is chatGPTs explanation of the code step by step:


GenerateMask takes one worm's measurements (a centerline + one width click), plus head click(s), and converts that into binary masks for:

- the full worm (worm_mask)
- head region (head_mask)
- tail region (tail_mask)
- a "neck" band near the head-body boundary (neck_mask)

plus boundary layers + bbox + a resampled/oriented centerline C + a diagnostics struct out

Below is what it does, in order, following the actual code in GenerateMask.m.


## 1) Read inputs + set default parameters

If params isn't provided or is missing fields, it fills defaults like:

- resample_step_px (default 1.5) - spacing used to resample the centerline
- morph_close_radius (default 2) - radius for morphological closing
- hole_fill (default true) - whether to imfill holes
- head_frac (default 0.18) - fraction of body length used as "head region"
- neck_halfwidth_frac (default 0.03) - half-width around the head boundary used for "neck"
- end_min_frac (default 0.12) - minimum tip diameter as fraction of max width
- taper_head, taper_tail, taper_shape - control how width tapers to the ends
- max_head_dist (default Inf) - optional gating: head click must be within this distance of an endpoint

This is done via the helper defaults(params, struct(...)).


## 2) Parse the "measures row" into centerline + max width

measures_row can be either:

- a cell: {centerline Nx2, width_pts 2x2}

or a struct with fields:

- .centerline (Nx2)
- .width_pts (2x2)
- optional .width_profile

parse_measures_row:

- converts those to a consistent Nx2 format (ensure_xy)
- forces width_pts to be exactly 2 points
- computes maxW = Euclidean distance between the two width points
  (this is treated as the worm's "maximum width" scale)


## 3) Resample the centerline to uniform-ish spacing

resample_centerline(C0, step):

- computes cumulative arc length along the polyline
- interpolates points every step pixels

returns:

- C: the resampled centerline (Mx2)
- s: normalized arc coordinate in [0,1] along the centerline

This makes later geometry (normals, polygons) more stable.


## 4) Pick which endpoint is the head (using head click(s))

pick_head_for_worm(C, head_coords, max_head_dist):

If head_coords is one point [x y]:

- it decides whether that point is closer to C(1,:) or C(end,:)
- sets is_start = true if closer to the start endpoint
- optionally errors if it's farther than max_head_dist from both endpoints

If head_coords is a Kx2 list:

- it chooses the head candidate that best matches an endpoint (nearest)
- returns which candidate index it used (head_idx)

It stores into out:

- out.picked_head
- out.is_start
- out.paired_head_idx


## 5) Orient the centerline so index 1 is always the head

If the head was closer to the end of the polyline, it flips it:

```matlab
if ~is_start

    C = flipud(C);

    s = flipud(1 - s);

end
```


## 6) Choose the “peak width” location (widest point along the worm)

(Currently set to mid point)

```matlab
idx_peak = ceil(length(s)/2);
``` 

It stores: 

- out.idx_peak 
- out.s_peak 
- out.maxW 


## 7) Build a width-per-point profile along the centerline

It constructs w_px (length M), a width in pixels at each centerline point.

Three cases: 

If measures_row included a .width_profile:

- it resamples that profile to length M (resample_profile) 
- normalizes it so the maximum is 1 
- scales by maxW 
- enforces a minimum tip width using end_min_frac 

Else if params.mean_worm exists: 

- it uses profile_from_template(...) (external helper) to create a profile 
- still scaled by maxW and constrained at the ends 

Else:

- it uses asymmetric_width_profile(...) (external helper) 
- uses different taper strength for head vs tail (taper_head vs taper_tail) 
- shape controlled by taper_shape (e.g., 'poly') 

Result: w_px is the worm diameter (or “thickness”) along the body. 


## 8) Convert centerline + widths into a polygon, then into a mask

centerline_to_polygon(C, w_px): 

- estimates a tangent direction at each point 
- converts it to a normal vector
- offsets the centerline by ± normal * (w_px/2) to get a left and right edge 
- builds a closed polygon by concatenating left edge + reversed right edge 

Then: 

```matlab
worm_mask = poly2mask(poly(:,1), poly(:,2), H, W);
``` 

Then postprocess:

- optionally imclose with a disk of radius morph_close_radius 
- optionally imfill(...,'holes') 
- ensures logical output 

So worm_mask is your final “full worm” binary segmentation. 


## 9) Create head / neck / tail masks by slicing the centerline

Let M = size(C,1). 

Head region 

- It picks an index:
  idx_head_end = round(P.head_frac * (M-1)); 
  (clamped to at least 3 and at most M)
- Then builds a polygon only from C(1:idx_head_end,:) (and matching widths), and masks it:  
  head_mask = poly2mask(...) 
- ANDs it with worm_mask (so it can’t leak outside) 
- Then:
  head_boundary = GetEndBoundary(head_mask) 
  (2-pixel-ish boundary band: inner perimeter OR outer perimeter) 

Neck region (band around the head boundary)

- It makes a band centered around idx_head_end: 
  neck_half = round(P.neck_halfwidth_frac * (M-1)) (at least 2) 
  [idx_neck_lo, idx_neck_hi] around the head boundary 
- It builds neck_poly from that slice and masks it. 
- Then it also builds a similar band near the tail side (it calls it tail_neck_poly) and ORs it in: 
  neck_mask = neck_mask | neck_tail_mask; 
- So neck_mask ends up being two “bands”: one near the head boundary and one near the tail-side symmetric location.

Tail region

- It takes the last idx_head_end points (same length as head slice):
  tail_poly = centerline_to_polygon(C((M-idx_head_end):end,:), ...) 
- tail_mask = poly2mask(...) 
- tail_mask = tail_mask & worm_mask; 
- tail_boundary = GetEndBoundary(tail_mask); 


## 10) Compute bounding box of the worm mask

stats = regionprops(worm_mask,'BoundingBox'); 

bbox = stats(1).BoundingBox  (or [0 0 0 0] if empty) 


## 11) Outputs you get back

- worm_mask : full body mask 
- head_mask : head segment mask (first head_frac of body) 
- tail_mask : tail segment mask (last head_frac of body) 
- neck_mask : band(s) around head boundary (+ a symmetric tail-side band)
- head_boundary, tail_boundary : “inner+outer” perimeter band masks 
- bbox : bounding box from regionprops 
- C : resampled, head-oriented centerline 
- out : diagnostics:
  - which head point was used + whether start was the head 
  - index of “peak” + max width, etc. 
