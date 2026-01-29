function [head_coords, canceled] = MarkHeadsInteractive(im, measures, head_coords, opts)
% Mark heads only if head_coords is empty. Otherwise, return as-is (headless).
%
% Inputs
%   im         : image matrix (H×W or H×W×3)
%   measures   : N×2 cell; {centerline Nx2, width_pts 2×2} per row
%   head_coords: [] or K×2 existing heads (if non-empty, UI is skipped)
%   opts       : (optional) struct:
%                  .expected_count : default size(measures,1)
%                  .save_path      : optional .mat path to save head_coords
%                  .title          : window title string
%                  .point_size     : default 36
%
% Outputs
%   head_coords : K×2 [x y] (empty if canceled and nothing clicked)
%   canceled    : true if user hit ESC/closed the window before confirming

if nargin < 4, opts = struct; end
if ~exist('head_coords','var') || isempty(head_coords)
    [head_coords, canceled] = do_interactive(im, measures, opts);
else
    canceled = false;  % nothing to do
end

% optional save
if ~isempty(head_coords) && isfield(opts,'save_path') && ~isempty(opts.save_path)
    try
        save(opts.save_path, 'head_coords');
    catch ME
        warning('Could not save head_coords to %s: %s', opts.save_path, ME.message);
    end
end
end

% ====================== internal ======================
function [pts, canceled] = do_interactive(im, measures, opts)
pts = zeros(0,2);
canceled = false;

exp_count  = get_opt(opts,'expected_count', size(measures,1));
pt_size    = get_opt(opts,'point_size', 36);
win_title  = get_opt(opts,'title', 'Click HEADS: left-click to add, U=undo, R=reset, Enter/right-click=finish, ESC=cancel');

% figure & axes
fig = figure('Name', win_title, 'NumberTitle','off', 'Color','w', ...
             'KeyPressFcn', @on_key, 'WindowButtonDownFcn', @on_click);
ax = axes('Parent', fig); 
im = rescale(im).*255;
try
    imagesc(im, 'Parent', ax); hold(ax,'on');
catch
    image(ax, im); axis(ax,'image'); set(ax,'YDir','reverse'); hold(ax,'on');
end

% overlay skeletons (thin cyan), show ends as small markers
N = size(measures,1);
for i=1:N
    C = ensure_xy(measures{i,1});
    plot(ax, C(:,1), C(:,2), '-', 'Color',[0 1 1 0.6], 'LineWidth', 1.0);
    if ~isempty(C)
        plot(ax, C(1,1),  C(1,2),  'o', 'MarkerSize',3, 'MarkerEdgeColor',[0 0.8 1]);
        plot(ax, C(end,1),C(end,2),'o', 'MarkerSize',3, 'MarkerEdgeColor',[0 0.8 1]);
    end
end
title(ax, sprintf('%s\nExpected heads: %d', win_title, exp_count), 'Interpreter','none');

% live scatter + counter UI
sc = scatter(ax, nan, nan, pt_size, 'r', 'filled', 'MarkerFaceAlpha', 0.9);
counter_txt = annotation(fig,'textbox',[0.01 0.01 0.35 0.07], ...
    'String', counter_str(), 'EdgeColor','none', 'Color',[0.1 0.1 0.1], ...
    'FontSize', 11, 'Interpreter','none');

% event loop (blocking until finish/cancel)
uiwait(fig);

% nested helpers -------------
    function on_click(~,~)
        if ~ishandle(ax), return; end
        sel = get(fig,'SelectionType');
        switch sel
            case 'normal'  % left-click -> add point
                cp = ax.CurrentPoint;  % 2×3; take first row XY
                xy = cp(1,1:2);
                if inside_image(xy, size(im))
                    pts(end+1,:) = xy; 
                    refresh_scatter();
                end
            case 'alt'     % right-click -> finish
                finish_and_close();
            case 'extend'  % middle-click -> finish as well (optional)
                finish_and_close();
            otherwise
                % 'open' = double-click -> finish
                finish_and_close();
        end
    end

    function on_key(~, evt)
        if ~isfield(evt,'Key'), return; end
        switch lower(evt.Key)
            case {'return','enter'}
                finish_and_close();
            case 'escape'
                canceled = isempty(pts);  % only flag canceled if nothing was confirmed
                close_safely();
            case 'u'  % undo last
                if ~isempty(pts)
                    pts(end,:) = [];
                    refresh_scatter();
                end
            case 'r'  % reset all
                pts = zeros(0,2);
                refresh_scatter();
        end
    end

    function finish_and_close()
    
        close_safely();
    end

    function refresh_scatter()
        if isempty(pts)
            set(sc,'XData', nan, 'YData', nan);
        else
            set(sc,'XData', pts(:,1), 'YData', pts(:,2));
        end
        if isgraphics(counter_txt)
            counter_txt.String = counter_str();
        end
        drawnow;
    end

    function s = counter_str()
        s = sprintf('Heads: %d  (expected: %d)\nU=undo, R=reset, Enter/Right-click=finish, ESC=cancel', ...
            size(pts,1), exp_count);
    end

    function tf = inside_image(xy, sz)
        H = sz(1); W = sz(2);
        tf = xy(1) >= 1 && xy(1) <= W && xy(2) >= 1 && xy(2) <= H;
    end

    function close_safely()
        if isvalid(fig)
            uiresume(fig);
            delete(fig);
        end
    end
end

% ------------- utilities -------------
function val = get_opt(opts, name, default)
if isfield(opts, name) && ~isempty(opts.(name))
    val = opts.(name);
else
    val = default;
end
end

function C = ensure_xy(A)
A = double(A);
if isempty(A), C = zeros(0,2); return, end
if size(A,2)==2
    C = A;
elseif size(A,1)==2
    C = A.';
else
    error('Expected Nx2 or 2×N for centerline.');
end
C = C(all(isfinite(C),2),:);
end

