function [varargout] = RWL1_DF_largescale(varargin)

% [vid_coef_rwcs, vid_recon_rwcs, vid_rMSE_rwcs, vid_PSNR_rwcs, vid_time_rwcs, vid_nProd_rwcs] = ...
%        RWL1_DF_largescale(MEAS_SIG, MEAS_FUN, DYN_FUN, DWTfunc, ...
%        param_vals, TRUE_VID)
%
%   The inputs are:
% 
% MEAS_SIG:   Mx1xT array of the measurements for the video frames
% MEAS_FUN:   Tx1 or 1x1 cell array of the measurement functions
% DYN_FUN:    Tx1 or 1x1 cell array of the dynamics functions
% DWTfunc:    Wavelet transform (sparsity basis)
% param_vals: struct of parameter values (has fields: lambda_val (tradeoff
%             parameter for BPDN), rwl1_mult (multiplicative parameter for
%             the re-weighting), rwl1_reg (additive value in denominator of
%             the re-weighting), beta (multiplicative value in denominator 
%             of the re-weighting) and tol (tolerance for TFOCS solver))
% TRUE_VID:   Sqrt(N)xSqrt(N)xT array of the true video sequence (optional,
%             to evaluate errors)
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
if isfield(param_vals, 'rwl1_reg')
    rwl1_reg = param_vals.rwl1_reg;
else
    rwl1_reg = 0.2;
end
if isfield(param_vals, 'rwl1_mult')
    rwl1_mult = param_vals.rwl1_mult;
else
    rwl1_mult = 0.4;
end
if isfield(param_vals, 'beta')
    w1 = param_vals.beta;
else
    w1 = 1;
end
if isfield(param_vals, 'tol')
    TOL = param_vals.tol;
else
    TOL = 0.001;
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

meas_func = MEAS_FUN{1};
Phi  = meas_func.Phi;
Phit = meas_func.Phit; 

num_frames = size(MEAS_SIG, 3);
% opts.tol = TOL;
% opts.printEvery = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run RWL1 on each frame

% Initialize outputs
vid_coef_drw = zeros(N2,1,num_frames);
vid_recon_drw = zeros(N,N,num_frames);
if rMSE_calc_opt
    vid_rMSE_drw = zeros(num_frames,1);
    vid_PSNR_drw = zeros(num_frames,1);
end
vid_time_drw = zeros(num_frames,1);
if count_nProd
    vid_nProd_drw = zeros(num_frames,1);
end

% Run first time-step
kk = 1;
tic

nProd_count = 0;
weights = 1;
for nn = 1:10
    % M Step: Solve BPDN
    if count_nProd
        Af = @(arg) apply_and_count(@(x) Phi(DWT_invert(x./weights)), arg);
        Ab = @(arg) apply_and_count(@(x) DWT_apply(Phit(x))./weights, arg);
    else
        Af = @(x) Phi(DWT_invert(x./weights));
        Ab = @(x) DWT_apply(Phit(x))./weights;
    end
%     A = linop_handles([M, N2], Af, Ab, 'R2R');
%     res = solver_L1RLS( A, reshape(MEAS_SIG(:, :, 1), [], 1), lambda_val, zeros(N2, 1), opts );
    res = SpaRSA(MEAS_SIG(:, :, 1), Af, lambda_val, 'ToleranceA', TOL,...
        'AT', Ab, 'verbose', 0, 'Initialization', 0, 'StopCriterion', 3 );
    res = res./weights;
    % E-step: Update weights
    weights = 1./(abs(real(res)) + rwl1_reg);

    if rMSE_calc_opt
        im_res = DWT_invert(res);
        disp([mean(weights), var(weights)])
        temp_rMSE = sum(sum((im_res - TRUE_VID(:, :, 1)).^2))/sum(sum(TRUE_VID(:, :, 1).^2));
        fprintf('Finished RW iteration %d, rMSE = %f.. \n', nn, temp_rMSE)
    else
        fprintf('Finished RW iteration %d.\n', nn)
    end
end

% Save reconstruction results
vid_coef_drw(:, :, 1) = res;
vid_recon_drw(:, :, 1) = im_res;
if rMSE_calc_opt
    % Save results
    vid_rMSE_drw(1) = sum(sum((vid_recon_drw(:, :, kk) - TRUE_VID(:, :, kk)).^2))/sum(sum(TRUE_VID(:, :, kk).^2));
    vid_PSNR_drw(1) = psnr(vid_recon_drw(:, :, 1), TRUE_VID(:, :, 1));
    TIME_ITER = toc;
    fprintf('Finished frame %d of %d in %f seconds. PSNR is %f. rMSE is %f. \n', 1, num_frames, TIME_ITER, vid_PSNR_drw(1), vid_rMSE_drw(1))
else
    TIME_ITER = toc;
    fprintf('Finished frame %d of %d in %f seconds.\n', 1, num_frames, TIME_ITER)
end
if count_nProd
    vid_nProd_drw(1) = nProd_count;
    fprintf('nProd is %d Ops.\n', vid_nProd_drw(1))
end
vid_time_drw(1) = TIME_ITER;

% Solve the remaining frames

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
elseif num_meas_func == num_frames
    dif_meas_func = 1;
else
    error('You need either the same dynamics function for all time or one dynamics function per time-step!')
end

for kk = 2:num_frames
    nProd_count = 0;
    
    % Get the Dynamics function
    if dif_dyn_func == 1
        f_dyn = DYN_FUN{kk};
    end
    % Get the Measurement function
    if dif_meas_func == 1
        meas_func = MEAS_FUN{kk};
        Phi  = meas_func.Phi;
        Phit = meas_func.Phit;
    end
    
    tic
    % Predict weights
    x_pred = DWT_apply(f_dyn(vid_recon_drw(:, :, kk-1)));
    
    % Initialize EM weights
    weights = 1;
    
    for nn = 1:10
        % M-Step: Solve BPDN
        if count_nProd
            Af = @(arg) apply_and_count(@(x) Phi(DWT_invert(x./weights)), arg);
            Ab = @(arg) apply_and_count(@(x) DWT_apply(Phit(x))./weights, arg);
        else
            Af = @(x) Phi(DWT_invert(x./weights));
            Ab = @(x) DWT_apply(Phit(x))./weights;
        end
%         A = linop_handles([M, N2], Af, Ab, 'R2R');
%         res = solver_L1RLS( A, MEAS_SIG(:, :, kk), lambda_val, zeros(N2, 1), opts );
        res = SpaRSA(MEAS_SIG(:, :, kk), Af, lambda_val, 'ToleranceA', TOL,...
            'AT', Ab, 'verbose', 0, 'Initialization', 0, 'StopCriterion', 3 );
        res = res./weights;
        % E-Step: Update weights
        weights = rwl1_mult./(w1*abs(real(res)) + abs(x_pred) + rwl1_reg);
    
        if rMSE_calc_opt
            im_res = DWT_invert(res);
            temp_rMSE = sum(sum((im_res - TRUE_VID(:, :, kk)).^2))/sum(sum(TRUE_VID(:, :, kk).^2));
            fprintf('Finished RW iteration %d, rMSE = %f.. \n', nn, temp_rMSE)
        else
            fprintf('Finished RW iteration %d.\n', nn)
        end
    end
    
    if rMSE_calc_opt
        vid_coef_drw(:, :, kk) = res;
        vid_recon_drw(:, :, kk) = im_res;
        vid_rMSE_drw(kk) = sum(sum((vid_recon_drw(:, :, kk) - TRUE_VID(:, :, kk)).^2))/sum(sum(TRUE_VID(:, :, kk).^2));
        vid_PSNR_drw(kk) = psnr(real(vid_recon_drw(:, :, kk)), TRUE_VID(:, :, kk));
        TIME_ITER = toc;
        fprintf('Finished frame %d of %d in %f seconds. PSNR is %f. rMSE is %f. \n', kk, num_frames, TIME_ITER, vid_PSNR_drw(kk), vid_rMSE_drw(kk))
    else
        im_res = DWT_invert(res);
        vid_coef_drw(:, :, kk) = res;
        vid_recon_drw(:, :, kk) = im_res;
        TIME_ITER = toc;
        fprintf('Finished frame %d of %d in %f seconds.\n', kk, num_frames, TIME_ITER)
    end
    if count_nProd
        vid_nProd_drw(kk) = nProd_count;
        fprintf('nProd is %d Ops.\n', vid_nProd_drw(kk))
    end
    vid_time_drw(kk) = TIME_ITER;
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set ouptputs

varargout = cell(nargout);
if rMSE_calc_opt
    if nargout > 0
        varargout{1} = vid_coef_drw;
    end
    if nargout > 1
        varargout{2} = vid_recon_drw;
    end
    if nargout > 2
        varargout{3} = vid_rMSE_drw;
    end
    if nargout > 3
        varargout{4} = vid_PSNR_drw;
    end
    if nargout > 4
        varargout{5} = vid_time_drw;
    end
    if nargout > 5
        varargout{6} = vid_nProd_drw;
    end
    if nargout > 6
        for kk = 7:nargout
            varargout{kk} = [];
        end
    end
else
    if nargout > 0
        varargout{1} = vid_coef_drw;
    end
    if nargout > 1
        varargout{2} = vid_recon_drw;
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
