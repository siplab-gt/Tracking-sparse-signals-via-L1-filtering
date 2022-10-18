function [varargout] = dcsamp_video(varargin)

% [vid_coef_dcsamp, vid_recon_dcsamp, vid_rMSE_dcsamp, vid_PSNR_dcsamp, vid_time_dcsamp, vid_nProd_dcsamp] = ...
%           BPDN_video(MEAS_SIG, MEAS_FUN, RunOptions, Params, DWTfunc, TRUE_VID)
%
%   The inputs are:
% 
% MEAS_SIG:   Mx1xT array of the measurements for the video frames
% MEAS_FUN:   Tx1 or 1x1 cell array of the measurement functions
% RunOptions: Structure
% Params:     Structure
% DWTfunc:    Wavelet transform (sparsity basis)
% TRUE_VID:   Sqrt(N)xSqrt(N)xT array of the true video sequence (optional,
%             to evaluate errors)
% 
%    The outputs are:
% 
% vid_coef_dcsamp:  Nx1xT array of inferred sparse coefficients
% vid_recon_dcsamp: Sqrt(N)xSqrt(N)xT array of the recovered video sequence
% vid_rMSE_dcsamp:  Tx1 array of rMSE values for the recovered video
% vid_PSNR_dcsamp:  Tx1 array of PSNR values for the recovered video
% vid_time_dcsamp:  Tx1 array of time values for the recovered video
% vid_nProd_dcsamp:  Tx1 array of number of products by A and At for the recovered video
% 
% Code by Aurèle Balavoine
% Department of Electrical and Computer Engineering,
% Georgia Institute of Technology
% 
% Last updated October 10, 2013. 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parse Inputs
MEAS_SIG = varargin{1};
MEAS_FUN = varargin{2};
RunOptions = varargin{3};
Params = varargin{4};
DWTfunc = varargin{5};

if nargin > 5
    rMSE_calc_opt = 1;
    TRUE_VID = varargin{6};
else
    rMSE_calc_opt = 0;
end

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

MEAS_FUN_DCS = cell(size(MEAS_SIG, 3), 1);
y_DCS = cell(size(MEAS_SIG, 3), 1);
            
if numel(MEAS_FUN) == 1
    meas_func = MEAS_FUN{1};
    Phi  = meas_func.Phi;
    Phit = meas_func.Phit;
    for kk = 1:size(MEAS_SIG, 3)
        if count_nProd
            Af = @(arg) apply_and_count(@(x) Phi(DWTfunc.invert(x)), arg);
            Ab = @(arg) apply_and_count(@(x) DWTfunc.apply(Phit(x)), arg);
        else
            Af = @(x) Phi(DWTfunc.invert(x));
            Ab = @(x) DWTfunc.apply(Phit(x));
        end
        MEAS_FUN_DCS{kk} = @(x,mode) handle_conversion(x,Af, Ab, mode); 
        y_DCS{kk} = MEAS_SIG(:, :, kk);
    end
else
    for kk = 1:size(MEAS_SIG, 3)
        meas_func = MEAS_FUN{kk};
        Phi  = meas_func.Phi;
        Phit = meas_func.Phit;
        if count_nProd
            Af = @(arg) apply_and_count(@(x) Phi(DWTfunc.invert(x)), arg);
            Ab = @(arg) apply_and_count(@(x) DWTfunc.apply(Phit(x)), arg);
        else
            Af = @(x) Phi(DWTfunc.invert(x));
            Ab = @(x) DWTfunc.apply(Phit(x));
        end
        MEAS_FUN_DCS{kk} =  @(x,mode) handle_conversion(x,Af, Ab, mode);
        y_DCS{kk} = MEAS_SIG(:, :, kk);
    end
end

num_frames = size(MEAS_SIG, 3);
% opts.tol = TOL;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run sp_frame_fxn

% Initialize outputs
vid_coef_dcsamp = zeros(N2,1,num_frames);
vid_recon_dcsamp = zeros(N,N,num_frames);
if rMSE_calc_opt
    vid_rMSE_dcsamp = zeros(num_frames,1);
    vid_PSNR_dcsamp = zeros(num_frames,1);
end

% DCS_AMP
[x_hat, vid_time_dcsamp, vid_nProd_dcsamp] = sp_multi_frame_fxn_mod(y_DCS, MEAS_FUN_DCS, Params, RunOptions);

for kk = 1:num_frames
    res = x_hat{kk};    
    im_res = DWT_invert(res);
    % Save reconstruction results
    vid_coef_dcsamp(:, :, kk) = res;
    vid_recon_dcsamp(:, :, kk) = im_res;
    if rMSE_calc_opt
        vid_rMSE_dcsamp(kk) = sum(sum((vid_recon_dcsamp(:, :, kk) - TRUE_VID(:, :, kk)).^2))/sum(sum(TRUE_VID(:, :, kk).^2));
        vid_PSNR_dcsamp(kk) = psnr(real(vid_recon_dcsamp(:, :, kk)), TRUE_VID(:, :, kk));
        fprintf('Finished frame %d of %d in %f seconds. PSNR is %f. rMSE is %f. \n', kk, num_frames, vid_time_dcsamp(kk), vid_PSNR_dcsamp(kk), vid_rMSE_dcsamp(kk))
    else
        fprintf('Finished frame %d of %d in %f seconds.\n', kk, num_frames, vid_time_dcsamp(kk))
    end
    if count_nProd
        fprintf('nProd is %d Ops.\n', vid_nProd_dcsamp(kk))
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set ouptputs

varargout = cell(nargout);
if rMSE_calc_opt
    if nargout > 0
        varargout{1} = vid_coef_dcsamp;
    end
    if nargout > 1
        varargout{2} = vid_recon_dcsamp;
    end
    if nargout > 2
        varargout{3} = vid_rMSE_dcsamp;
    end
    if nargout > 3
        varargout{4} = vid_PSNR_dcsamp;
    end
    if nargout > 4
        varargout{5} = vid_time_dcsamp;
    end
    if nargout > 5
        varargout{6} = vid_nProd_dcsamp;
    end
    if nargout > 6
        for kk = 7:nargout
            varargout{kk} = [];
        end
    end
else
    if nargout > 0
        varargout{1} = vid_coef_dcsamp;
    end
    if nargout > 1
        varargout{2} = vid_recon_dcsamp;
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
