function [varargout] = BPDN_video(varargin)

% [vid_coef_cs, vid_recon_cs, vid_rMSE_cs, vid_PSNR_cs, vid_time_cs, vid_nProd_cs] = ...
%           BPDN_video(MEAS_SIG, MEAS_FUN, lambda_val, TOL, DWTfunc, TRUE_VID)
%
%   The inputs are:
% 
% MEAS_SIG:   Mx1xT array of the measurements for the video frames
% MEAS_FUN:   Tx1 or 1x1 cell array of the measurement functions
% lambda_val: Scalar value for the BPDN sparsity tradeoff
% TOL:        Scalar value for the tolerance in the TFOCS solver
% DWTfunc:    Wavelet transform (sparsity basis)
% TRUE_VID:   Sqrt(N)xSqrt(N)xT array of the true video sequence (optional,
%             to evaluate errors)
% 
%    The outputs are:
% 
% vid_coef_cs:  Nx1xT array of inferred sparse coefficients
% vid_recon_cs: Sqrt(N)xSqrt(N)xT array of the recovered video sequence
% vid_rMSE_cs:  Tx1 array of rMSE values for the recovered video
% vid_PSNR_cs:  Tx1 array of PSNR values for the recovered video
% vid_time_cs:  Tx1 array of time values for the recovered video
% vid_nProd_cs:  Tx1 array of number of products by A and At for the recovered video
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
lambda_val = varargin{3};
TOL = varargin{4};
DWTfunc = varargin{5};

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

% Set up A and At for TFOCS
if count_nProd
    Af = @(arg) apply_and_count(@(x) Phi(DWT_invert(x)), arg);
    Ab = @(arg) apply_and_count(@(x) DWT_apply(Phit(x)), arg);
else
    Af = @(x) Phi(DWT_invert(x));
    Ab = @(x) DWT_apply(Phit(x));
end
A = linop_handles([M, N2], Af, Ab, 'R2R');

num_frames = size(MEAS_SIG, 3);
opts.tol = TOL;
opts.printEvery = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run BPDN on each frame

% Initialize outputs
vid_coef_cs = zeros(N2,1,num_frames);
vid_recon_cs = zeros(N,N,num_frames);
if rMSE_calc_opt
    vid_rMSE_cs = zeros(num_frames,1);
    vid_PSNR_cs = zeros(num_frames,1);
end
vid_time_cs = zeros(num_frames,1);
if count_nProd
%     opts.countOps = 1;
    vid_nProd_cs = zeros(num_frames,1);
end

% Set up the measurement function
num_meas_func = numel(MEAS_FUN);    
if num_meas_func == 1
    dif_func = 0;
elseif num_meas_func == num_frames
    dif_func = 1;
else
    error('You need either the same dynamics function for all time or one dynamics function per time-step!')
end

res = zeros(N2, 1);
for kk = 1:num_frames
    nProd_count = 0;
    
    % Set up the measurement function if different for each frame
    if (dif_func == 1)&&(kk>1)
        meas_func = MEAS_FUN{kk};
        Phi  = meas_func.Phi;
        Phit = meas_func.Phit;
        % Set up A and At for TFOCS
        if count_nProd
            Af = @(arg) apply_and_count(@(x) Phi(DWT_invert(x)), arg);
            Ab = @(arg) apply_and_count(@(x) DWT_apply(Phit(x)), arg);
        else
            Af = @(x) Phi(DWT_invert(x));
            Ab = @(x) DWT_apply(Phit(x));
        end
        A = linop_handles([M, N2], Af, Ab, 'R2R');
    end
    
    % Solve the BPDN objective
    tic
    res = solver_L1RLS( A, MEAS_SIG(:, :, kk), lambda_val, res, opts );
%     [res, data] = solver_L1RLS( A, MEAS_SIG(:, :, kk), lambda_val, zeros(N2, 1), opts );
    im_res = DWT_invert(res);
    
    % Save reconstruction results
    vid_coef_cs(:, :, kk) = res;
    vid_recon_cs(:, :, kk) = im_res;
    if rMSE_calc_opt
        vid_rMSE_cs(kk) = sum(sum((vid_recon_cs(:, :, kk) - TRUE_VID(:, :, kk)).^2))/sum(sum(TRUE_VID(:, :, kk).^2));
        vid_PSNR_cs(kk) = psnr(real(vid_recon_cs(:, :, kk)), TRUE_VID(:, :, kk));
        TIME_ITER = toc;
        fprintf('Finished frame %d of %d in %f seconds. PSNR is %f. rMSE is %f. \n', kk, num_frames, TIME_ITER, vid_PSNR_cs(kk), vid_rMSE_cs(kk))
    else
        TIME_ITER = toc;
        fprintf('Finished frame %d of %d in %f seconds.\n', kk, num_frames, TIME_ITER)
    end
    if count_nProd
        vid_nProd_cs(kk) = nProd_count; %data.counts(end,3);
        fprintf('nProd is %d Ops.\n', vid_nProd_cs(kk))
    end
    vid_time_cs(kk) = TIME_ITER;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set ouptputs

varargout = cell(nargout);
if rMSE_calc_opt
    if nargout > 0
        varargout{1} = vid_coef_cs;
    end
    if nargout > 1
        varargout{2} = vid_recon_cs;
    end
    if nargout > 2
        varargout{3} = vid_rMSE_cs;
    end
    if nargout > 3
        varargout{4} = vid_PSNR_cs;
    end
    if nargout > 4
        varargout{5} = vid_time_cs;
    end
    if nargout > 5
        varargout{6} = vid_nProd_cs;
    end
    if nargout > 6
        for kk = 7:nargout
            varargout{kk} = [];
        end
    end
else
    if nargout > 0
        varargout{1} = vid_coef_cs;
    end
    if nargout > 1
        varargout{2} = vid_recon_cs;
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
