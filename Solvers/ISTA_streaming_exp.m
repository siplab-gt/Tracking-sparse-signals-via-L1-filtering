function [varargout] = ISTA_streaming_exp(varargin)

% [coef, coefCost] = ...
%           ISTA_streaming_exp(MEAS_SIG, MEAS_FUN, PARAMS, COST_FUN)
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
% COST_FUN:   coefficient cost function (optional, default=[])
% 
%    The outputs are:
% 
% coef:  NxT array of inferred sparse coefficients
% coefCost : 1x(Txniter) array of output cost
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

temp = Phit(MEAS_SIG(:, 1));
N = numel(temp);
clear temp

num_frames = size(MEAS_SIG, 2);

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
    niter = 10;
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
if (nargin > 3) && (nargout>1)
    cost_calc = 1;
    coefCost = zeros(1,niter*num_frames);
    COST_FUN = varargin{4};
else
    cost_calc = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run LCA on each frame

% Initialize outputs
coef = zeros(N,num_frames);

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
    % Set up the measurement function if different for each frame
    if (dif_func == 1)&&(kk>1)
        meas_func = MEAS_FUN{kk};
        Phi  = meas_func.Phi;
        Phit = meas_func.Phit;
    end
    if lambda_decay
        lambda = lambda_val(kk);
    end
    
    if cost_calc
        % Solve the BPDN objective
        [res, rMSEk,coefCostk] = solver_ISTA(MEAS_SIG(:, kk), Phi, Phit, lambda, ...
                    niter, tau, init, [], COST_FUN); % removes the tolerance stopping criteria
        % Save reconstruction results
        coef(:, kk) = res;
        init = res;
        % Store results
        coefCost((kk-1)*niter+1:kk*niter) = coefCostk;
    else
        % Solve the BPDN objective
        [res] = solver_ISTA(MEAS_SIG(:, kk), Phi, Phit, lambda, ...
                    niter, tau, init); % removes the tolerance stopping criteria
        % Save reconstruction results
        coef(:, kk) = res;
        init = res;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set ouptputs

if cost_calc
    if nargout > 0
        varargout{1} = coef;
    end
    if nargout > 1
        varargout{2} = coefCost;
    end
    if nargout > 2
        for kk = 6:nargout
            varargout{kk} = [];
        end
    end
elseif ~coef_calc
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
