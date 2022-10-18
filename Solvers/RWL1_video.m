function [varargout] = RWL1_video(varargin)

% [vid_coef_rwcs, vid_recon_rwcs, vid_rMSE_rwcs, vid_PSNR_rwcs, vid_time_rwcs, vid_nProd_rwcs] = ...
%           RWL1_video(MEAS_SIG, MEAS_SEL, lambda_val, TOL, DWTfunc, TRUE_VID)
%
%   The inputs are:
% 
% MEAS_SIG:   Mx1xT array of the measurements for the video frames
% MEAS_FUN:   Tx1 or 1x1 cell array of the measurement functions
% param_vals: 1x3 vector of parameter values
% TOL:        Scalar value for the tolerance in the TFOCS solver
% DWTfunc:    Wavelet transform (sparsity basis)
% TRUE_VID:   Sqrt(N)xSqrt(N)xT array of the true video sequence (optional,
%             to evaluate rMSE and PSNR)
% 
%    The outputs are:
% 
% vid_coef_rwcs:  Nx1xT array of inferred sparse coefficients
% vid_recon_rwcs: Sqrt(N)xSqrt(N)xT array of the recovered video sequence
% vid_rMSE_rwcs:  Tx1 array of rMSE values for the recovered video
% vid_PSNR_rwcs:  Tx1 array of PSNR values for the recovered video
% vid_time_rwcs:  Tx1 array of time values for the recovered video
% vid_nProd_rwcs:  Tx1 array of number of products by A and At for the recovered video
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
param_vals = varargin{3}; 
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

% M = numel(MEAS_SIG(:, :, 1));
temp = Phit(MEAS_SIG(:, :, 1));
N = sqrt(numel(temp));
N2 = numel(DWT_apply(temp));
clear temp

lambda_val = param_vals(1);
rwl1_reg = param_vals(2);
rwl1_mult = param_vals(3);

num_frames = size(MEAS_SIG, 3);
% opts.tol = TOL;
% opts.printEvery = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run RWL1 on each frame

% Initialize outputs
vid_coef_rwcs = zeros(N2,1,num_frames);
vid_recon_rwcs = zeros(N,N,num_frames);
if rMSE_calc_opt
    vid_rMSE_rwcs = zeros(num_frames,1);
    vid_PSNR_rwcs = zeros(num_frames,1);
end
vid_time_rwcs = zeros(num_frames,1);
if count_nProd
    vid_nProd_rwcs = zeros(num_frames,1);
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
    end
    
    tic
    weights = 1;
    for nn = 1:10
        % M-Step: Solve weighted BPDN
        if count_nProd
            Af = @(arg) apply_and_count(@(x) Phi(DWT_invert(x./weights)), arg);
            Ab = @(arg) apply_and_count(@(x) DWT_apply(Phit(x))./weights, arg);
        else
            Af = @(x) Phi(DWT_invert(x./weights));
            Ab = @(x) DWT_apply(Phit(x))./weights;
        end
%         A = linop_handles([M, N2], Af, Ab, 'R2R');
        
%         res = solver_L1RLS(A, MEAS_SIG(:, :, kk), lambda_val, zeros(N2, 1), opts);
        res = SpaRSA(MEAS_SIG(:, :, kk), Af, lambda_val, 'ToleranceA', TOL,...
            'AT', Ab, 'verbose', 0, 'Initialization', res, 'StopCriterion', 3 );
        res = res./weights;
        % E-stepL Reset the weights
        weights = rwl1_mult./(abs(res) + rwl1_reg);
        
        if rMSE_calc_opt
            im_res = DWT_invert(res);
            temp_rMSE = sum(sum((im_res - TRUE_VID(:, :, kk)).^2))/sum(sum(TRUE_VID(:, :, kk).^2));
            fprintf('Finished RW iteration %d. rMSE is %f.\n', nn, temp_rMSE)
        else
            fprintf('Finished RW iteration %d.\n', nn)
        end
    end
    
    if rMSE_calc_opt
        im_res = DWT_invert(res);
        % Save reconstruction results
        vid_coef_rwcs(:, :, kk) = res;
        vid_recon_rwcs(:, :, kk) = im_res;
        vid_rMSE_rwcs(kk) = sum(sum((vid_recon_rwcs(:, :, kk) - TRUE_VID(:, :, kk)).^2))/sum(sum(TRUE_VID(:, :, kk).^2));
        vid_PSNR_rwcs(kk) = psnr(real(vid_recon_rwcs(:, :, kk)), TRUE_VID(:, :, kk));
        TIME_ITER = toc;
        fprintf('Finished frame %d of %d in %f seconds. PSNR is %f. rMSE is %f. \n', kk, num_frames, TIME_ITER, vid_PSNR_rwcs(kk), vid_rMSE_rwcs(kk))
    else
        im_res = DWT_invert(res);
        % Save reconstruction results
        vid_coef_rwcs(:, :, kk) = res;
        vid_recon_rwcs(:, :, kk) = im_res;
    end
    if count_nProd
        vid_nProd_rwcs(kk) = nProd_count;
        fprintf('nProd is %d Ops.\n', vid_nProd_rwcs(kk))
    end
    vid_time_rwcs(kk) = TIME_ITER;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set ouptputs

varargout = cell(nargout);
if rMSE_calc_opt
    if nargout > 0
        varargout{1} = vid_coef_rwcs;
    end
    if nargout > 1
        varargout{2} = vid_recon_rwcs;
    end
    if nargout > 2
        varargout{3} = vid_rMSE_rwcs;
    end
    if nargout > 3
        varargout{4} = vid_PSNR_rwcs;
    end
    if nargout > 4
        varargout{5} = vid_time_rwcs;
    end
    if nargout > 5
        varargout{6} = vid_nProd_rwcs;
    end
    if nargout > 6
        for kk = 7:nargout
            varargout{kk} = [];
        end
    end
else
    if nargout > 0
        varargout{1} = vid_coef_rwcs;
    end
    if nargout > 1
        varargout{2} = vid_recon_rwcs;
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
