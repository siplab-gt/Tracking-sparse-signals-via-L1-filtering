
function [x, rMSE, coefCost] = solver_ISTA(y, A, At, lambda, niter, tau, initState,...
    xtrue, costFunc)

% Solver_ISTA: Iterative Soft-Thresholding Algorithm.
%
% Usage
%   [x, rMSE, coefCost, numCoef, trueRelErr] = solver_ISTA(y, A, At, ...
%       lambda, niter, tau, initState, xtrue, costFunc)
%
% Input
%       y               signal vector of length n
%       A               implicit (function handle)
%       At              implicit (function handle)
%       lambda          tradeoff parameter (threshold value)
%       initState       initial length-n state (u) vector of the LCA
%                           (default is length-n zero vector)
%       nIter         maximum number of iterations to perform (default=300)
%       tau             gradient stepsize (default=1))
%
% Outputs
%       x               length m vector solution of the LCA
%       u               length m vector state of the LCA (x=actFunc(u))
%
% Description
%   Implements the ISTA
%
%   u(n+1) = x(n) + tau*At*(y-A*x(n)) with x = actFunc(u, actFuncArgs)
%
%   to estimate the solution of  min ||A*x-y||^2 + lambda*sum(C(x_i)).
%
%   Modified by Aurèle Balavoine
%   Modification date: 10/28/2013


if (nargin<4)
    error('Must specify y, A, At, and lambda.');
end
if (nargin<7)
    initState = zeros(size(A,2));
end
if (nargin<6)
    tau=1;
end
if (nargin<5)
    niter=300;
end
if (nargin > 7) && (nargout > 1) && (~isempty(xtrue))
    rMSE_calc_opt = 1;
    rMSE = zeros(niter+1,1);
    rMSE(1) = sum((initState - xtrue).^2)/sum(xtrue.^2);
else
    rMSE_calc_opt = 0;
    rMSE = [];
end
if (nargin>8) && (nargout >2)
    cost_calc = 1;
    coefCost = zeros(1,niter);
else
    cost_calc = 0;
end

%put the threshold in terms of function handle 'curLambda' so we don't have to
%care whether it was a scalar, vector or cell array
if  isscalar(lambda)
    curLambda = @(k) lambda;
elseif isvector(actFuncArgs)
    curLambda = @(k) lambda(k);
else
    curLambda = @(k) lambda(:,k);
end

if (length(lambda) > 1)
    niter = min([niter length(lambda)]);
end

%set up first iteration
% u = initState;						%state variables
% x =  sthresh(u, curLambda(1));					%output (activation) variables
x = initState;
recon = A(x);			%current reconstruction
resid = y-recon;					%current residual
newproj = At(resid);	%projection of residual onto dictionary

contIter = 1;						%continue iterating?
countIter = 1;						%iteration counter

while contIter
  
  %find new states/coefficients
  u = x + tau*newproj;
  x = sthresh(u, curLambda(countIter));
  recon = A(x);
  resid = y-recon;
  newproj = At(resid);

  if (countIter >= niter)
    contIter=0;
  end
  
  if rMSE_calc_opt
    rMSE(countIter+1) = sum((x - xtrue).^2)/sum(xtrue.^2);
  end
  
  if cost_calc
      coefCost(countIter) = costFunc(x, curLambda(countIter));
  end
  
  countIter = countIter+1;    

end
