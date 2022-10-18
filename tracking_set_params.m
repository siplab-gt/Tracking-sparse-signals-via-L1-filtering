TOL = 1e-3;            % TFOCS tolerance parameter
for alg = 1:num_algo
    algo_name = algo_names{alg};
    switch algo_name
        
        case 'BPDN'
            algo_params.BPDN.lambda_val = 0.05;     % threshold
            algo_params.BPDN.TOL = TOL;             % stop when norm_dx < TOL * max( norm_x, 1 )
            
        case 'GPSR'
            algo_params.GPSR.lambda = 0.05;         % threshold
            algo_params.GPSR.TOL = TOL;             % stop when LCP estimate of relative 
                                                    % distance to solution
                                                    % falls below TOL
            
        case 'SpaRSA'
            algo_params.SPARSA.lambda = 0.01;       % threshold
            algo_params.SPARSA.TOL = TOL;           % stop when LCP estimate of relative 
                                                    % distance to solution
                                                    % falls below TOL
            
        case 'RWL1'                                 % uses SpaRSA
            algo_params.RWL1.lambda_val = 0.11;     % threshold
            algo_params.RWL1.rwl1_reg = 0.1;        % rw-regularizer
            algo_params.RWL1.rwl1_mult = 0.2;       % weights
            algo_params.RWL1.TOL = TOL;             % stop when LCP estimate of relative 
                                                    % distance to solution
                                                    % falls below TOL
            
        case 'BPDN-DF'                              % uses SpaRSA
            algo_params.BPDNDF.DYN_FUN = DYN_FUN;   % dynamics function
            algo_params.BPDNDF.lambda_val = 0.01; 	% threshold
            algo_params.BPDNDF.lambda_history = 0.2;% tradeoff parameter
                                                    % between prediction and data fidelity
            algo_params.BPDNDF.TOL = TOL;           % stop when LCP estimate of relative 
                                                    % distance to solution
                                                    % falls below TOL
            
        case 'RWL1-DF'                              % uses SpaRSA
            algo_params.RWL1DF.DYN_FUN = DYN_FUN;   % dynamics function
            algo_params.RWL1DF.lambda_val = 0.003; 	% threshold
            algo_params.RWL1DF.rwl1_reg = 1;        % rw-regularizer (+)
            algo_params.RWL1DF.rwl1_mult = 4;       % weights
            algo_params.RWL1DF.beta = 1;            % rw-regularizer (x)
            algo_params.RWL1DF.tol = TOL;           % stop when LCP estimate of relative 
                                                    % distance to solution
                                                    % falls below TOL
            
        case 'LCA'
            algo_params.LCA.lambda = 0.05;          % threshold
            algo_params.LCA.tau = 1;                % gradient-step \eta
            algo_params.LCA.niter = 1;              % # iterations P
            
        case 'ISTA P1'
            algo_params.ISTAP1.lambda = 0.05;       % threshold
            algo_params.ISTAP1.tau = 2;             % gradient-step \eta
            algo_params.ISTAP1.niter = 3;           % # iterations P
            
        case 'ISTA P2'
            algo_params.ISTAP2.lambda = 0.05;       % threshold
            algo_params.ISTAP2.tau = 2;             % gradient-step \eta
            algo_params.ISTAP2.niter = 10;          % # iterations P
            
        case 'Homotopy'
            algo_params.l1h.tau = 0.5;              % threshold
            algo_params.l1h.tau_decay = 0;          % decrease the threshold
            algo_params.l1h.delx_mode = 'mil';
            algo_params.l1h.maxiter = 500;
            
        case {'DCS-AMP'}
            DCSAMPOptions = struct('eq_iter', []);
            % Number of equalization iterations to perform,
            % per forward/backward pass, at each timestep [dflt: 25] [try:10]
            DCSAMPOptions.eq_iter = 25; 
            % Type of BP algorithm to use during equalization 
            % at each timestep: (1) for standard BP, (2) for AMP [dflt: 2]
            DCSAMPOptions.alg = 2; 
            % Number of fwd/bwd passes (-1 to filter)
            DCSAMPOptions.smooth_iter = -1; 
            % Update DCS-AMP parameters using EM learning
            % No available in filter mode.
            DCSAMPOptions.update = 0; 
            % List of parameter to update
            DCSAMPOptions.update_list = {'lambda', 'p01', 'p10', 'eta',...
                'kappa', 'alpha', 'rho', 'sig2e'};
            % tau-thresholding parameter [deflt=0.25]
            DCSAMPOptions.tau = 0.0015; 
            % Print msgs
            DCSAMPOptions.verbose = 0; 
            % "Squelch" parameter for inactives [dflt=1e-7]
            DCSAMPOptions.eps = 1e-5;
            % Scalar variance of the additive white Gaussian noise
            % [dflt=1e-2] [fd1 = 1e-4] [fd2 = 0.035606]
            DCSAMPParams.sig2e = 0.0042; % 0.001;
            % prior probability of non-zero tap [dflt=0.04] [fd1 = 0.045]
            % [fd2 = 0.678525]
            DCSAMPParams.lambda = 0.0438; % 0.1;
            % Pr{s_n(t) = 0 | s_n(t-1) = 1} [dflt=0.05] [fd1 = 0.007] [fd2
            % = 0.025320]
            DCSAMPParams.p01 = 0.0257; % 0.04;
            % Pr{s_n(t) = 1 | s_n(t-1) = 0} [dflt=lambda*p01/(1 -lambda)]
            % no need to set if because dependent on p01
%             DCSAMPParams.p10 = DCSAMPParams.lambda*...
%                 DCSAMPParams.p01/(1 - DCSAMPParams.p01);
            % Complex mean of active coefficients [dflt=0]
            DCSAMPParams.eta = 0;
            % Circular variance of active coefficients	[dflt=1][try=1e4]
            % [fd1 = 0.7] [fd2 = 0.062632]
            DCSAMPParams.kappa = 0.91; % 1;
            % Innovation rate of thetas (1 = total) [dflt=0.10] [fd1 =
            % 0.03] [fd2 = 0.001411]
            DCSAMPParams.alpha = 0.0945; % 0.1;
            % Driving noise variance [dflt=(2 - alpha)*kappa/alpha] [try=1e5] [fd1 = 25]
            % no need to set it because dependent variable
%             DCSAMPParams.rho = (2 - DCSAMPParams.alpha)*...
%                DCSAMPParams.kappa/DCSAMPParams.alpha; 
%             DCSAMPParams.rho = DCSAMPParams.kappa / DCSAMPParams.alpha^2; %25;
            % save options
            algo_params.DCSAMP.RunOptions = DCSAMPOptions;
            algo_params.DCSAMP.Params = DCSAMPParams;
    end
end