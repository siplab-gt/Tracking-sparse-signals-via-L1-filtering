function [varargout] = BPDN_DF_video(varargin)

% [vid_coef_dcs, vid_recon_dcs, vid_rMSE_dcs, vid_PSNR_dcs, vid_time_dcs, vid_nProd_dcs] = ...
%        BPDN_DF_video(MEAS_SIG, MEAS_SEL, DYN_FUN, DWTfunc, param_vals, ...
%        TRUE_VID)
%
%   The inputs are:
% 
% MEAS_SIG:   Mx1xT array of the measurements for the video frames
% MEAS_FUN:   Tx1 or 1x1 cell array of the measurement functions
% DYN_FUN:    Tx1 or 1x1 cell array of the dynamics functions
% DWTfunc:    Wavelet transform (sparsity basis)
% param_vals: struct of parameter values (has fields: lambda_val (tradeoff
%             parameter for BPDN), lambda_history (tradeoff parameter
%             between prediction and data fidelity), and tol (tolerance for
%             TFOCS solver)) 
% TRUE_VID:   Sqrt(N)xSqrt(N)xT array of the true video sequence (optional,
%             to evaluate errors)
% 
%    The outputs are:
% 
% vid_coef_dcs:  Nx1xT array of inferred sparse coefficients
% vid_recon_dcs: Sqrt(N)xSqrt(N)xT array of the recovered video sequence
% vid_rMSE_dcs:  Tx1 array of rMSE values for the recovered video
% vid_PSNR_dcs:  Tx1 array of PSNR values for the recovered video
% vid_time_dcs:  Tx1 array of time values for the recovered video
% vid_nProd_dcs:  Tx1 array of number of products by A and At for the recovered video
% 
% 
% Code by Adam Charles, 
% Department of Electrical and Computer Engineering,
% Georgia Institute of Technology
%
% Edited by Aurèle Balavoine
% Department of Electrical and Computer Engineering,
% Georgia Institute of Technology
% 
% Last updated October 10, 2013. 
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parse Inputs
MEAS_SIG = varargin{1};
MEAS_FUN = varargin{2};
DYN_FUN = varargin{3};
DWTfunc = varargin{4};
param_vals = varargin{5};

if nargin > 5
    rMSE_calc_opt = 1;
    TRUE_VID = varargin{6};
else
    rMSE_calc_opt = 0;
end

global nProd_count
if nargout > 5
    count_nProd = 1;
else
    count_nProd = 0;
end

if isfield(param_vals, 'lambda_val')
    lambda_val = param_vals.lambda_val;
else
    lambda_val = 0.001;
end
if isfield(param_vals, 'lambda_history')
    lambda_history = param_vals.lambda_history;
else
    lambda_history = 0.2;
end
if isfield(param_vals, 'tol')
    TOL = param_vals.tol;
else
    TOL = 0.01;
end

DWT_apply = DWTfunc.apply;
DWT_invert = DWTfunc.invert;

meas_func = MEAS_FUN{1};
Phi  = meas_func.Phi;
Phit = meas_func.Phit; 

M = numel(MEAS_SIG(:, :, 1));
temp = Phit(MEAS_SIG(:, :, 1));
N = sqrt(numel(temp));
N2 = numel(DWT_apply(temp));
clear temp

num_frames = size(MEAS_SIG, 3);
% opts.tol = TOL;
% opts.printEvery = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve for initial frame

% Initialize outputs
vid_coef_dcs = zeros(N2,1,num_frames);
vid_recon_dcs = zeros(N,N,num_frames);
if rMSE_calc_opt
    vid_rMSE_dcs = zeros(num_frames,1);
    vid_PSNR_dcs = zeros(num_frames,1);
end
vid_time_dcs = zeros(num_frames,1);
if count_nProd
    vid_nProd_dcs = zeros(num_frames,1);
end

if count_nProd
    Af = @(arg) apply_and_count(@(x) Phi(DWT_invert(x)), arg);
    Ab = @(arg) apply_and_count(@(x) DWT_apply(Phit(x)), arg);
else
    Af = @(x) Phi(DWT_invert(x));
    Ab = @(x) DWT_apply(Phit(x));
end
% A = linop_handles([M, N2], Af, Ab, 'R2R');
nProd_count = 0;

tic
% res = solver_L1RLS(A, MEAS_SIG(:, :, 1), lambda_val, zeros(N2, 1), opts );
res = SpaRSA(MEAS_SIG(:, :, 1), Af, lambda_val, 'ToleranceA', TOL,...
        'AT', Ab, 'verbose', 0, 'Initialization', 0, 'StopCriterion', 3 );
im_res = DWT_invert(res);

% Save reconstruction results
vid_coef_dcs(:, :, 1) = res;
vid_recon_dcs(:, :, 1) = im_res;
if rMSE_calc_opt
    vid_rMSE_dcs(1) = sum(sum((vid_recon_dcs(:, :, 1) - TRUE_VID(:, :, 1)).^2))/sum(sum(TRUE_VID(:, :, 1).^2));
    vid_PSNR_dcs(1) = psnr(real(vid_recon_dcs(:, :, 1)), TRUE_VID(:, :, 1));
    TIME_ITER = toc;
    fprintf('Finished frame %d of %d in %f seconds. PSNR is %f. rMSE is %f. \n', 1, num_frames, TIME_ITER, vid_PSNR_dcs(1), vid_rMSE_dcs(1))
else
    TIME_ITER = toc;
    fprintf('Finished frame %d of %d in %f seconds. \n', 1, num_frames, TIME_ITER)
end
if count_nProd
    vid_nProd_dcs(1) = nProd_count;
    fprintf('nProd is %d Ops.\n', vid_nProd_dcs(1))
end
vid_time_dcs(1) = TIME_ITER;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve for rest of frames

% Set up the dynamics function
num_dyn_func = numel(DYN_FUN);    
if num_dyn_func == 1
    dif_dyn_func = 0;
    f_dyn = DYN_FUN{1};
elseif num_dyn_func == num_frames
    dif_dyn_func = 1;
else
    error('You need either the same dynamics function for all time or one dynamics function per time-step!')
end

% Set up the measurement function
num_meas_func = numel(MEAS_FUN);    
if num_meas_func == 1
    dif_meas_func = 0;
    if count_nProd
        Af = @(arg) apply_and_count(@(x) [lambda_history*x; Phi(DWT_invert(x))], arg);
        Ab = @(arg) apply_and_count(@(x) (DWT_apply(Phit(x(N2+1:end))) + lambda_history*x(1:N2)), arg);
    else
        Af = @(x) [lambda_history*x; Phi(DWT_invert(x))];
        Ab = @(x) DWT_apply(Phit(x(N2+1:end))) + lambda_history*x(1:N2);
    end
%     A = linop_handles([M+N2, N2], Af, Ab, 'R2R');
elseif num_meas_func == num_frames
    dif_meas_func = 1;
else
    error('You need either the same dynamics function for all time or one dynamics function per time-step!')
end

for kk = 2:num_frames
    nProd_count = 0;
    
    % Get the Dynamics function
    if dif_dyn_func
        f_dyn = DYN_FUN{kk};
    end
    % Get the Measurement function
    if dif_meas_func
        meas_func = MEAS_FUN{kk};
        % Set  up A and At for TFOCS
        Phi  = meas_func.Phi;
        Phit = meas_func.Phit; 
        if count_nProd
            Af = @(arg) apply_and_count(@(x) [lambda_history*x; Phi(DWT_invert(x))], arg);
            Ab = @(arg) apply_and_count(@(x) (DWT_apply(Phit(x(N2+1:end))) + lambda_history*x(1:N2)), arg);
        else
            Af = @(x) [lambda_history*x; Phi(DWT_invert(x))];
            Ab = @(x) DWT_apply(Phit(x(N2+1:end))) + lambda_history*x(1:N2);
        end
%         A = linop_handles([M+N2, N2], Af, Ab, 'R2R');
    end
    
    tic
    % Calculate state prediction
    x_pred = DWT_apply(f_dyn(vid_recon_dcs(:, :, kk-1)));
    
    % Optimize the BPDN objective function with TFOCS
%     res = solver_L1RLS( A, [lambda_history*x_pred; ...
%         MEAS_SIG(:, :, kk)], lambda_val, res, opts );
    % Optimize the BPDN objective function with SpaRSA
    res = SpaRSA([lambda_history*x_pred; MEAS_SIG(:, :, kk)], Af, lambda_val, 'ToleranceA', TOL,...
        'AT', Ab, 'verbose', 0, 'Initialization', res, 'StopCriterion', 3 );
    im_res = DWT_invert(res);

    % Save reconstruction results
    vid_coef_dcs(:, :, kk) = res;
    vid_recon_dcs(:, :, kk) = im_res;
    if rMSE_calc_opt
        vid_rMSE_dcs(kk) = sum(sum((vid_recon_dcs(:, :, kk) - TRUE_VID(:, :, kk)).^2))/sum(sum(TRUE_VID(:, :, kk).^2));
        vid_PSNR_dcs(kk) = psnr(real(vid_recon_dcs(:, :, kk)), TRUE_VID(:, :, kk));
        TIME_ITER = toc;
        fprintf('Finished frame %d of %d in %f seconds. PSNR is %f. rMSE is %f. \n', kk, num_frames, TIME_ITER, vid_PSNR_dcs(kk), vid_rMSE_dcs(kk))
    else
        TIME_ITER = toc;
        fprintf('Finished frame %d of %d in %f seconds.\n', kk, num_frames, TIME_ITER)
    end
    if count_nProd
        vid_nProd_dcs(kk) = nProd_count;
        fprintf('nProd is %d Ops.\n', vid_nProd_dcs(kk))
    end
    vid_time_dcs(kk) = TIME_ITER;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set ouptputs

varargout = cell(nargout);
if rMSE_calc_opt
    if nargout > 0
        varargout{1} = vid_coef_dcs;
    end
    if nargout > 1
        varargout{2} = vid_recon_dcs;
    end
    if nargout > 2
        varargout{3} = vid_rMSE_dcs;
    end
    if nargout > 3
        varargout{4} = vid_PSNR_dcs;
    end
    if nargout > 4
        varargout{5} = vid_time_dcs;
    end
    if nargout > 5
        varargout{6} = vid_nProd_dcs;
    end
    if nargout > 6
        for kk = 7:nargout
            varargout{kk} = [];
        end
    end
else
    if nargout > 0
        varargout{1} = vid_coef_dcs;
    end
    if nargout > 1
        varargout{2} = vid_recon_dcs;
    end
    if nargout > 2
        for kk = 3:nargout
            varargout{kk} = [];
        end
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
