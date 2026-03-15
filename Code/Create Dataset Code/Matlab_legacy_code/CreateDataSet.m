mean_worm = [0.18,0.4,0.55,0.68,0.77,0.83,0.89,0.92, 0.95,0.97,0.98,0.99,...
    0.99,0.98,0.96,0.93,0.88,0.81,0.72,0.61,0.51,0.4,0.31,0.17];

params.mean_worm.to_head= (mean_worm(1:12));
params.mean_worm.to_tail= (mean_worm(13:end));
opts = struct;
params.interactive   = false;             % click heads only if missing
params.save_format   = 'mat';            % or 'tiff' / 'mat'
params.max_head_dist = inf;
params.max_end_frac  = 0.40;
params.lambda_end    = 1.2;
params.neck_halfwidth_frac = 0.035;
params.head_frac = 0.15;
opts.params        = params;  


save_training_images('path\to\marked\pics','path\to\output', opts);
