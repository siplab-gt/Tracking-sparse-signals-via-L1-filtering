function [varargout] = ISTA_streaming(varargin)

% [coef, rMSE, PSNR, time, rMSEt] = ...
%           ISTA_streaming(MEAS_SIG, MEAS_FUN, PARAMS, TRUE_SIG)
%
%   The inputs are:
% 
% MEAS_SIG:   MxT array of the measurements for the signal
% MEAS_FUN:   Tx1 or 1x1 cell array of the measurement functions
% PARAMS:     contains the parameters for ISTA:
%   - lambda: scalar value for the BPDN sparsity tradeoff
%   - niter:  scalar value for the number of iterations of ISTA per timestep
%   - tau:    scalar value for the gradient stepsize
%   - init:   Nx1 array of initial coefficient values
% TRUE_SIG:   NxT array of the true signal sequence (optional,
%             to evaluate errors)
% 
%    The outputs are:
% 
% coef:  NxT array of inferred sparse coefficients
% rMSE:  (T+1)x1 array of rMSE values for the timestep
% PSNR:  Tx1 array of PSNR values for the timestep
% time:  Tx1 array of ISTA time values for the timestep
% rMSEt: (niter x T +1)x1 array of rMSE values for each ISTA iteration
% 
% code by Aurèle Balavoine
% Department of Electrical and Computer Engineering,
% Georgia Institute of Technology
% 
% Last updated October 28, 2013. 
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parse Inputs
MEAS_SIG = varargin{1};
MEAS_FUN = varargin{2};
PARAMS = varargin{3};

meas_func = MEAS_FUN{1};
Phi  = meas_func.Phi;
Phit = meas_func.Phit;

% M = numel(MEAS_SIG(:, :, 1));
temp = Phit(MEAS_SIG(:, 1));
N = numel(temp);
clear temp
% A = @(x) Phi(DWT_invert(x));
% At = @(x) DWT_apply(Phit(x));

num_frames = size(MEAS_SIG, 2);

if nargin > 3
    rMSE_calc_opt = 1;
    TRUE_SIG = varargin{4};
else
    rMSE_calc_opt = 0;
    TRUE_SIG = [];
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
if isfield(PARAMS,'niter')
    niter = PARAMS.niter;
else
    niter = 100;
end
if isfield(PARAMS,'tau')
    tau = PARAMS.tau;
else
    tau = 1;
end
if isfield(PARAMS,'init')
    init = PARAMS.init;
else
    init = zeros(N,1);
end
if ~isfield(PARAMS,'verbose')
    PARAMS.verbose = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run ISTA on each frame

% Initialize outputs
coef = zeros(N,num_frames);
if rMSE_calc_opt
    rMSE = zeros(num_frames+1,1);
    PSNR = zeros(num_frames,1);
    rMSEt = zeros(niter*num_frames+1,1);
end
time = zeros(num_frames,1);

% Set up the measurement function
num_meas_func = numel(MEAS_FUN);    
if num_meas_func == 1
    dif_func = 0;
elseif num_meas_func == num_frames
    dif_func = 1;
else
    error('You need either the same dynamics function for all time or one dynamics function per time-step!')
end

if rMSE_calc_opt
    rMSEt(1) = sum(sum((init - TRUE_SIG(:, 1)).^2))/sum(sum(TRUE_SIG(:, 1).^2));
    rMSE(1) = rMSEt(1);
end

for kk = 1:num_frames
    tic
    % Set up the measurement function if different for each frame
    if (dif_func == 1)&&(kk>1)
        meas_func = MEAS_FUN{kk};
        Phi  = meas_func.Phi;
        Phit = meas_func.Phit;
%         A = @(x) Phi(DWT_invert(x));
%         At = @(x) DWT_apply(Phit(x));
    end
    if lambda_decay
        lambda = lambda_val(kk);
    end
    
    if rMSE_calc_opt
        % Solve the BPDN objective
        [res, rMSEk] = solver_ISTA(MEAS_SIG(:, kk), Phi, Phit, lambda, ...
                    niter, tau, init, TRUE_SIG(:, kk)); % removes the tolerance stopping criteria
        % Save reconstruction results
        coef(:, kk) = res;
        init = res;
        % Store results
        rMSE(kk+1) = sum(sum((res - TRUE_SIG(:, kk)).^2))/sum(sum(TRUE_SIG(:, kk).^2));
        PSNR(kk) = psnr(res, TRUE_SIG(:, kk));
        rMSEt((kk-1)*niter+2:kk*niter+1) = rMSEk(2:end);
        TIME_ITER = toc;
        if PARAMS.verbose
            fprintf('Finished frame %d of %d in %f seconds. PSNR is %f. rMSE is %f. \n',...
                kk, num_frames, TIME_ITER, PSNR(kk), rMSE(kk))
        end
    else
        % Solve the BPDN objective
        [res] = solver_ISTA(MEAS_SIG(:, kk), Phi, Phit, lambda, ...
                    niter, tau, init); % removes the tolerance stopping criteria
        % Save reconstruction results
        coef(:, kk) = res;
        init = res;
        TIME_ITER = toc;
        if PARAMS.verbose
            fprintf('Finished frame %d of %d in %f seconds.\n', kk, num_frames, TIME_ITER)
        end
    end
    time(kk) = TIME_ITER;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set ouptputs

if rMSE_calc_opt
    if nargout > 0
        varargout{1} = coef;
    end
    if nargout > 1
        varargout{2} = rMSE;
    end
    if nargout > 2
        varargout{3} = PSNR;
    end
    if nargout > 3
        varargout{4} = time;
    end
    if nargout > 4
        varargout{5} = rMSEt;
    end
    if nargout > 5
        for kk = 6:nargout
            varargout{kk} = [];
        end
    end
elseif (rMSE_calc_opt ~= 1)
    if nargout > 0
        varargout{1} = coef;
    end
    if nargout > 1
        for kk = 2:nargout
            varargout{kk} = [];
        end
    end
else
    error('How did you get here?')
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
