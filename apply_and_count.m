
% APPLY_AND_COUNT	Internal routine.
%	A wrapper function used to facilitate the counting of multiplications
%	by A and At.

function varargout = apply_and_count( func, varargin )

global nProd_count

[ varargout{1:max(nargout,1)} ] = func( varargin{:} );
nProd_count = nProd_count + 1;