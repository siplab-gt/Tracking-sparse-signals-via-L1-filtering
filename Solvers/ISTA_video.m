function [varargout] = ISTA_video(varargin)

% [vid_coef_ista, vid_recon_ista, vid_rMSE_ista, vid_PSNR_ista, vid_time_ista, vid_nProd_ista] = ...
%           ISTA_video(MEAS_SIG, MEAS_FUN, PARAMS, DWTfunc, TRUE_VID)
%
%   The inputs are:
% 
% MEAS_SIG:   Mx1xT array of the measurements for the video frames
% MEAS_FUN:   Tx1 or 1x1 cell array of the measurement functions
% PARAMS:          contains the parameters for ISTA:
%   - lambda: scalar value for the BPDN sparsity tradeoff
%   - niter:  scalar value for the number of iterations of ISTA per
%             timestep (P)
%   - tau:    scalar value for the gradient stepsize (eta)
%   - init:   Nx1 array of initial coefficient values
% DWTfunc:    Wavelet transform (sparsity basis)
% TRUE_VID:   Sqrt(N)xSqrt(N)xT array of the true video sequence (optional,
%             to evaluate errors)
% 
%    The outputs are:
% 
% vid_coef_ista:  Nx1xT array of inferred sparse coefficients
% vid_recon_ista: Sqrt(N)xSqrt(N)xT array of the recovered video sequence
% vid_rMSE_ista:  Tx1 array of rMSE values for the recovered video
% vid_PSNR_ista:  Tx1 array of PSNR values for the recovered video
% vid_time_ista:  Tx1 array of time values for the recovered video
% vid_nProd_ista:  Tx1 array of number of products by A and At for the recovered video
% 
% code by Aurèle Balavoine
% Department of Electrical and Computer Engineering,
% Georgia Institute of Technology
% 
% Last updated October 31, 2013. 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parse Inputs
MEAS_SIG = varargin{1};
MEAS_FUN = varargin{2};
PARAMS = varargin{3};
DWTfunc = varargin{4};

if nargin > 4
    rMSE_calc_opt = 1;
    TRUE_VID = varargin{5};
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

if count_nProd
    Af = @(arg) apply_and_count(@(x) Phi(DWT_invert(x)), arg);
    Ab = @(arg) apply_and_count(@(x) DWT_apply(Phit(x)), arg);
else
    Af = @(x) Phi(DWT_invert(x));
    Ab = @(x) DWT_apply(Phit(x));
end

% M = numel(MEAS_SIG(:, :, 1));
temp = Phit(MEAS_SIG(:, 1));
N = sqrt(numel(temp));
N2 = numel(DWT_apply(temp));
clear temp
num_frames = size(MEAS_SIG, 3);

if isfield(PARAMS,'init')
    res = PARAMS.init;
else
    res = zeros(N2,1);
end
if isfield(PARAMS,'niter')
    niter = PARAMS.niter;
else
    niter = 10;
end
if isfield(PARAMS,'tau')
    tau = PARAMS.tau;
else
    tau = 1/norm(A(At(eye(N2))));
end
if isfield(PARAMS,'lambda')
    lambda_val = PARAMS.lambda;
else
    fprintf(['error: you must provide a tradeoff value lambda in the' ...
        ' structure array PARAMS.\n'])
end
if length(lambda_val)>1
    lambda_decay = 1;
else
    lambda_decay = 0;
    lambda = lambda_val;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run ISTA on each frame

% Initialize outputs
vid_coef_ista = zeros(N2,1,num_frames);
vid_recon_ista = zeros(N,N,num_frames);
if rMSE_calc_opt
    vid_rMSE_ista = zeros(num_frames,1);
    vid_PSNR_ista = zeros(num_frames,1);
end
vid_time_ista = zeros(num_frames,1);
if count_nProd
    vid_nProd_ista = zeros(num_frames,1);
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

for kk = 1:num_frames
    nProd_count = 0;
    
    % Set up the measurement function if different for each frame
    if (dif_func == 1)&&(kk>1)
        meas_func = MEAS_FUN{kk};
        Phi  = meas_func.Phi;
        Phit = meas_func.Phit;
        if count_nProd
            Af = @(arg) apply_and_count(@(x) Phi(DWT_invert(x)), arg);
            Ab = @(arg) apply_and_count(@(x) DWT_apply(Phit(x)), arg);
        else
            Af = @(x) Phi(DWT_invert(x));
            Ab = @(x) DWT_apply(Phit(x));
        end
    end
    if lambda_decay
        lambda = lambda_val(kk);
    end
    
    tic
    % Solve the BPDN objective
    res = solver_ISTA(MEAS_SIG(:,:, kk), Af, Ab, lambda, niter, tau, res); 
    im_res = DWT_invert(res);
    
    % Save reconstruction results
    vid_coef_ista(:, :, kk) = res;
    vid_recon_ista(:, :, kk) = im_res;
    if rMSE_calc_opt
        vid_rMSE_ista(kk) = sum(sum((im_res - TRUE_VID(:,:, kk)).^2))/sum(sum(TRUE_VID(:,:, kk).^2));
        vid_PSNR_ista(kk) = psnr(real(im_res), TRUE_VID(:,:, kk));
        TIME_ITER = toc;
        fprintf('Finished frame %d of %d in %f seconds. PSNR is %f. rMSE is %f. \n', kk, num_frames, TIME_ITER, vid_PSNR_ista(kk), vid_rMSE_ista(kk))
    else
        TIME_ITER = toc;
        fprintf('Finished frame %d of %d in %f seconds.\n', kk, num_frames, TIME_ITER)
    end
    if count_nProd
        vid_nProd_ista(kk) = nProd_count;
        fprintf('nProd is %d Ops.\n', vid_nProd_ista(kk))
    end
    vid_time_ista(kk) = TIME_ITER;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set ouptputs

varargout = cell(nargout);
if rMSE_calc_opt
    if nargout > 0
        varargout{1} = vid_coef_ista;
    end
    if nargout > 1
        varargout{2} = vid_recon_ista;
    end
    if nargout > 2
        varargout{3} = vid_rMSE_ista;
    end
    if nargout > 3
        varargout{4} = vid_PSNR_ista;
    end
    if nargout > 4
        varargout{5} = vid_time_ista;
    end
    if nargout > 5
        varargout{6} = vid_nProd_ista;
    end
    if nargout > 6
        for kk = 7:nargout
            varargout{kk} = [];
        end
    end
else
    if nargout > 0
        varargout{1} = vid_coef_ista;
    end
    if nargout > 1
        varargout{2} = vid_recon_ista;
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
