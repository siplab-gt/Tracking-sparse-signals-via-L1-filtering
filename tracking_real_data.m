%%
% This code generates Figure 4 in the paper "Iterative and Continuous-time
% Soft-Thresholding with a Dynamic Input" by Aurèle Balavoine, Christopher J.
% Rozell and Justin K. Romberg

% set plot parameters
set(0,'DefaultLineLineWidth',2); 
set(0,'DefaultLineMarkersize',10); 
set(0,'DefaultTextFontsize',16); 
set(0,'DefaultAxesFontsize',16); 
set(0,'defaultaxeslinewidth',1);

%% Add path to solvers, GPSR and SpaRSA

disp('Enter path to solvers (GPSR, SpaRSA, etc.) if used')
addpath Solvers
addpath C:\Users\abalavoine3\Work\GPSR_6.0
addpath C:\Users\abalavoine3\Work\SpaRSA_2.0
addpath C:\Users\abalavoine3\Work\SpaRSA_2.0\Measurements
% addpath C:\Users\abalavoine3\Dropbox\classes\LCA\Simulations\Solvers

%% Set experiment parameters

clearall = 0;

if clearall
    clear all
    clearall = 1;
end

saveres = 0;
startat = 1;

% To generate the same figures as the paper:
test_seed.rand_seed = 46010;
test_seed.randn_seed = 85551;
rand('seed',test_seed.rand_seed)
randn('seed',test_seed.randn_seed);

% % Choose video folder % %

foldername = 'Videos';

%% Setup Parameters

% % Running Options % %

reset_samp = 1;     % set to 1 to pick a new video sequence for each trial
frame_end = 300;    % last frame to select from the video
T_s = 40;           % Number of frames for each trial sequence
num_trials = 100;   % Number of trials to average
samp_factor = 0.25; % Subsampling ratio to determine number of measurements
noise_var = 0.0001; % Measurement Noise Variance
save_every = 10;    % Save the results every #save_every trials

% % Algorithms available for testing % %

algo_names = {'BPDN', 'GPSR', 'SpaRSA', 'BPDN-DF', 'RWL1-DF', 'LCA', 'ISTA P1',...
    'ISTA P2', 'DCS-AMP', 'RWL1'};
num_algo = numel(algo_names);   %  number of available algorithms
for num = 1:num_algo            % attribute a number to each algorithm
    algo = genvarname(algo_names{num});
    algo_number.(algo) = num;
end

% % Algorithms to use in the simulation % %

compute_algo = {'SpaRSA', 'BPDN-DF', 'RWL1-DF', 'DCS-AMP', 'LCA', 'ISTA P1',...
    'ISTA P2'};  % names of algorithms to use
num_compute = numel(compute_algo);              % number of algorithms to use

% % Save running options % %

algo_params.reset_samp = reset_samp;
algo_params.frame_end = frame_end;
for alg = 1:num_compute         % add the names of the new algorithms to use
    algo_name = compute_algo{alg};
    if ~isfield(algo_params,'compute_algo')
        algo_params.compute_algo = {algo_name};
    elseif ~ismember(algo_name,algo_params.compute_algo)
        algo_params.compute_algo = {algo_params.compute_algo{:},algo_name};
    end
end
algo_params.algo_names = algo_names;
algo_params.test_seed = test_seed;
algo_params.T_s = T_s;
algo_params.num_trials = num_trials;
algo_params.samp_factor = samp_factor;
algo_params.noise_var = noise_var;

% % Set and save the general parameters for the algorithms % %

% Set dynamics function to identity for BPDN-DF and RWL1-DF
f_dyn = @(z) z;
DYN_FUN{1} = f_dyn;

% % Set algorithms' parameters % %
tracking_set_params;

% % Set up the sparsifying matrix % %

% Uncomment to use 4 level Daubechies Wavelet Transform
% XFM = Wavelet('Daubechies',4,4);	
% DWTfunc.apply = @(q) reshape(XFM*q, [], 1);
% DWTfunc.invert = @(q) (XFM')*reshape(q, sqrt(N), sqrt(N));

% Uncomment to use 4 level Dual-Tree Discrete Wavelet Transforms
dwt_opts.J = 4;
DWTfunc.apply = @(q) DTDWT2D_mat(q, dwt_opts.J);
DWTfunc.invert = @(q) iDTDWT2D_mat(q, dwt_opts.J);

% % Initialize result arrays % %

if clearall % if clearing everything
    PSNR_ALL = zeros(T_s, num_algo, num_trials);
    rMSE_ALL = zeros(T_s, num_algo, num_trials);
    time_ALL = zeros(T_s, num_algo, num_trials);
    nProd_ALL = zeros(T_s, num_algo, num_trials);
    
elseif (~exist('rMSE_ALL','var')) % if the result arrays don't exist yet
    PSNR_ALL = zeros(T_s, num_algo, num_trials);
    rMSE_ALL = zeros(T_s, num_algo, num_trials);
    time_ALL = zeros(T_s, num_algo, num_trials);
    nProd_ALL = zeros(T_s, num_algo, num_trials);
    
elseif (size(rMSE_ALL,2)<num_algo) % if the result arrays are too short 
    nalgold = size(rMSE_ALL,2);     % (i.e. a new algorithm has been added)
    PSNR_temp = PSNR_ALL;
    rMSE_temp = rMSE_ALL;
    time_temp = time_ALL;
    nProd_temp = nProd_ALL;
    PSNR_ALL = zeros(T_s, num_algo, num_trials);
    rMSE_ALL = zeros(T_s, num_algo, num_trials);
    time_ALL = zeros(T_s, num_algo, num_trials);
    nProd_ALL = zeros(T_s, num_algo, num_trials);
    PSNR_ALL(:,1:nalgold,:) = PSNR_temp;
    rMSE_ALL(:,1:nalgold,:) = rMSE_temp;
    time_ALL(:,1:nalgold,:) = time_temp;
    nProd_ALL(:,1:nalgold,:) = nProd_temp;
    clear PSNR_temp PSNR_temp PSNR_temp nProd_temp nalgold
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run test

% display the current time
timenow = datestr(fix(clock),15);
disp(['--------------------------',datestr(now),'-------------------------------'])

% if the simulations have not been interrupted, startat = 1 and we use the
% current time in the name of the file to save the results in.
if startat==1
    savename = ['LCA_ISTA_video_' num2str(num_trials) '_' datestr(fix(clock),10) '_' ...
        datestr(fix(clock),5) '_' datestr(fix(clock),7) '_' timenow(1:2) '_' timenow(4:5) '.mat'];
end

for trial_index = 1:num_trials    
    
    disp(['-------------------------- trial # ',num2str(trial_index), ' of ', num2str(num_trials), ' -------------------------------'])
    
    if reset_samp
        
        % % Load video % %
        tracking_load_video; % return vid : ground truth

        % % Create measurements % %
             
        N1 = size(vid, 1);       % length of y-axis
        N2 = size(vid, 2);       % length of x-axis
        N = N1*N2;
        M = ceil(samp_factor*N);    % number of measurements

        vidmeas = zeros(M, 1, T_s);    % measurements
        MEAS_FUN = cell(1);         % contains the measurement matrix
        
%         for kk = 1:T_s            % some algorithms can take a varying Phi matrix
    
        % % Set up noiselets
        
        % Noise options
        q = randperm(N)';       % random permutation of 1:N
        OM = q(1:M);         % vector of random subset of integers in 1:N
        Phi  = @(z) A_noiselet (z, OM(:));           % A_noiselet: R^N -> R^M
        Phit = @(z) At_noiselet(z, OM(:), N);        % At_noiselet: R^M -> R^N
        meas_func.Phi = @(z) Phi(reshape(z, [], 1)); % Phi: R^{N1 x N2)} -> R^M
%         meas_func.Phit = @(z) reshape(Phit(z), sqrt(N), sqrt(N)); % Phit: R^M -> R^{sqrt(N)x sqrtN)}
        meas_func.Phit = @(z) reshape(Phit(z), N1, N2); % Phit: R^M -> R^{N1 x N2)}
        MEAS_FUN{1} = meas_func;
%         A = @(x) A_noiselet( reshape( iDTDWT2D_mat(x,dwt_opts.J),[],1), OM(:));
%         At = @(x) DTDWT2D_mat(reshape( At_noiselet(x,OM(:),N) ,sqrt(N),sqrt(N)),dwt_opts.J);
%         A = @(x) meas_func.Phi(DWTfunc.invert(x));
%         At = @(x) DWTfunc.apply(meas_func.Phit(x));
        for kk = 1:T_s
            vidmeas(:, :, kk) = meas_func.Phi(vid(:, :, kk)) + sqrt(noise_var)*randn(M, 1);
        end
%     end
    end
    
    % % Run algorithms % %
    
    if trial_index >= startat
        for alg=1:num_compute
            algo_name = compute_algo{alg};
            disp(['-------------------------- ', algo_name, ' - trial ',...
                num2str(trial_index), ' -------------------------------'])
            
            switch algo_name

        % BPDN Reconstruction
                case 'BPDN'                
                    lambda_val = algo_params.BPDN.lambda_val;
                    [vid_coef_cs, vid_recon_cs, vid_rMSE, vid_PSNR, vid_time, vid_nProd] = ...
                           BPDN_video(vidmeas, MEAS_FUN, lambda_val, TOL, DWTfunc, vid);

        % GPSR Reconstruction
                case 'GPSR'
                    lambda_val = algo_params.GPSR.lambda;
                    [vid_coef_gpsr, vid_recon_gpsr, vid_rMSE, vid_PSNR, vid_time, vid_nProd] = ...
                           GPSR_video(vidmeas, MEAS_FUN, lambda_val, TOL, DWTfunc, vid);

        % SPARSA Reconstruction
                case 'SpaRSA'
                    lambda_val = algo_params.SPARSA.lambda;
                    [vid_coef_gpsr, vid_recon_gpsr, vid_rMSE, vid_PSNR, vid_time, vid_nProd] = ...
                           SPARSA_video(vidmeas, MEAS_FUN, lambda_val, TOL, DWTfunc, vid);

        % RWL1 Reconstruction
                case 'RWL1'
                    [vid_coef_rwcs, vid_recon_rwcs, vid_rMSE, vid_PSNR, vid_time, vid_nProd] = ...
                          RWL1_video(vidmeas, MEAS_FUN, [algo_params.RWL1.lambda_val,...
                          algo_params.RWL1.rwl1_reg, algo_params.RWL1.rwl1_mult], ...
                          TOL, DWTfunc, vid);

        % BPDN-DF Reconstruction
                case 'BPDN-DF'
                    param_vals.lambda_val = algo_params.BPDNDF.lambda_val;
                    param_vals.lambda_history = algo_params.BPDNDF.lambda_history;
                    param_vals.tol = TOL;
                    [vid_coef_dcs, vid_recon_dcs, vid_rMSE, vid_PSNR, vid_time, vid_nProd] = ...
                        BPDN_DF_video(vidmeas, MEAS_FUN, DYN_FUN, DWTfunc, param_vals, vid);

        % RWL1-DF Reconstruction
                case 'RWL1-DF'
                    param_vals.lambda_val = algo_params.RWL1DF.lambda_val;
                    param_vals.rwl1_reg = algo_params.RWL1DF.rwl1_reg;
                    param_vals.rwl1_mult = algo_params.RWL1DF.rwl1_mult;
                    param_vals.beta = algo_params.RWL1DF.beta;
                    param_vals.tol = TOL;
                    [vid_coef_drw, vid_recon_drw, vid_rMSE, vid_PSNR, vid_time, vid_nProd] = ...
                        RWL1_DF_largescale(vidmeas, MEAS_FUN, DYN_FUN, DWTfunc, param_vals, vid);

        % LCA Reconstruction
                case 'LCA'
                    [vid_coef_lca, vid_recon_lca, vid_rMSE, ...
                        vid_PSNR, vid_time, vid_nProd] = ISTA_video(vidmeas, MEAS_FUN,...
                        algo_params.LCA, DWTfunc, vid);

        % ISTA Reconstruction
                case 'ISTA P1'
                    [vid_coef_istap1, vid_recon_istap1, vid_rMSE, ...
                        vid_PSNR, vid_time, vid_nProd] = ISTA_video(vidmeas, MEAS_FUN,...
                        algo_params.ISTAP1, DWTfunc, vid);

        % ISTA Reconstruction
                case 'ISTA P2'
                    [vid_coef_istap2, vid_recon_istap2, vid_rMSE, ...
                        vid_PSNR, vid_time, vid_nProd] = ISTA_video(vidmeas, MEAS_FUN,...
                        algo_params.ISTAP2, DWTfunc, vid);

        % DCS-AMP Reconstruction
                case 'DCS-AMP'
                    [vid_coef_dcsamp, vid_recon_dcsamp, vid_rMSE, vid_PSNR, vid_time, vid_nProd] = ...
                        dcsamp_video(vidmeas, MEAS_FUN, algo_params.DCSAMP.RunOptions,...
                        algo_params.DCSAMP.Params, DWTfunc, vid);

                otherwise
                    disp(['ERROR: The algorithm ', algo_name, ' is not available or not well-defined.'])
            end

            % % store results % %
            curr_id = algo_number.(genvarname(algo_name));
            PSNR_ALL(:, curr_id, trial_index) = vid_PSNR(:);
            rMSE_ALL(:, curr_id, trial_index) = vid_rMSE(:);
            time_ALL(:, curr_id, trial_index) = vid_time(:);
            nProd_ALL(:, curr_id, trial_index) = vid_nProd(:);
        end
        
        % % save temporary results % %
        if saveres && ~mod(trial_index, save_every)
            save(savename,'PSNR_ALL','rMSE_ALL','time_ALL','nProd_ALL','algo_params');
        end
    end
    
end
% % save results % %
if saveres
    save(savename,'PSNR_ALL','rMSE_ALL','time_ALL','nProd_ALL','algo_params');
end

disp(['--------------------------',datestr(now),'-------------------------------'])

%% Plot

foldername = 'C:\Users\abalavoine3\Work'; % lab path

% % Algorithm to be plotted % %
plot_algo = {'SpaRSA', 'BPDN-DF', 'RWL1-DF', 'DCS-AMP', 'LCA', 'ISTA P1',...
    'ISTA P2'};
num_plot = numel(plot_algo);    % number of algorithms to plot

% algorithms's names
algo_names = algo_params.algo_names;
num_algo = numel(algo_names);
for num = 1:num_algo
    algo = genvarname(algo_names{num});
    algo_number.(algo) = num;
end
% rename
algo_names{algo_number.(genvarname('ISTA P1'))} = ['ISTA (P=' num2str(algo_params.ISTAP1.niter)...
    ', \eta=' num2str(algo_params.ISTAP1.tau) ')'];
algo_names{algo_number.(genvarname('ISTA P2'))} = ['ISTA (P=' num2str(algo_params.ISTAP2.niter)...
    ', \eta=' num2str(algo_params.ISTAP2.tau) ')'];
algo_names{algo_number.(genvarname('LCA'))} = ['ISTA (P=' num2str(algo_params.LCA.niter)...
    ', \eta=' num2str(algo_params.LCA.tau) ')'];
% algorithms actually computed
% compute_algo = algo_params.compute_algo;

% % Style of lines for figure % %

% Color
cmap = flipud(hot(num_plot+2));
color_style_list = cell(1,num_plot);
for kk = 1:num_plot
    color_style_list{kk} = cmap(kk+2,:);
end
color_style_list{2} = [0,0,0];
line_style_list = {'-', '--', '-.', ':', '-'};
line_style_list2 = {'.', '.', '.', '.', '.'};
line_width_list = [6,4,3*ones(1,num_algo-3),1];
markersize_list = [1,1,1,1,3];
marker_list = {'none','none','none','none','d'};

%% Figure 4 (a): plot the mean rMSE

plot_handle = zeros(num_plot,1);


if exist(foldername,'dir')
    figure('units','pixels','Position',[50 50 780 600]);
else
    figure('units','pixels','Position',[50 50 680 500]);
end

for kk = 1:num_plot
    alg = algo_number.(genvarname(plot_algo{kk}));
    plot_handle(kk) = semilogy(mean(rMSE_ALL(:, alg,:), 3), 'color',color_style_list{kk},'linestyle',...
        line_style_list{mod(kk-1,5)+1}, 'LineWidth', line_width_list(kk),...
        'MarkerSize',markersize_list(mod(kk-1,5)+1), 'Marker',marker_list{mod(kk-1,5)+1},...
        'MarkerFaceColor',color_style_list{kk},...
        'DisplayName',algo_names{alg});
    hold on;
end
box on
if exist(foldername,'dir')
    set(gca, 'FontSize', 18, 'Xlim', [1,40],...
    'units','pixels','Position',[110 75 650 500]);
else
    set(gca, 'FontSize', 18, 'Xlim', [1,40],...
    'units','pixels','Position',[90 65 570 410]);
end

ylim([0.035 0.8]) 
xlabel('Frame Number', 'FontSize', 22)
ylabel('Mean rMSE', 'FontSize', 22)
% Split legend
leg2 = legend(plot_handle(5:end),'location', 'northeast');
set(leg2,'box','off');
ah2=axes('units','pixels','position',get(gca,'position'),...
    'FontSize', 18, 'visible','off');
leg2pos = get(leg2,'position');
leg1 = legend(ah2,plot_handle(1:4),'location', 'northwest');
set(leg1,'box','off');
leg1pos = get(leg1,'position');
set(leg1,'position',[leg2pos(1)-leg1pos(3),leg1pos(2:4)]);
annotation('rectangle','units','pixels',...
    'Position',[leg2pos(1)-leg1pos(3),leg1pos(2),...
    leg1pos(3)+leg2pos(3),max(leg1pos(4),leg2pos(4))],...
    'linewidth',0.5);

%% Figure 4 (b): plot the mean NPROD

plot_handle = zeros(num_plot,1);

if exist(foldername,'dir')
    figure('units','pixels','Position',[50 50 780 600]);
else
    figure('units','pixels','Position',[50 50 680 500]);
end
for kk = 1:num_plot
    alg = algo_number.(genvarname(plot_algo{kk}));
    plot_handle(kk) = semilogy(mean(nProd_ALL(:, alg,:), 3), 'color',color_style_list{kk},'linestyle',...
        line_style_list{mod(kk-1,5)+1}, 'LineWidth', line_width_list(kk),...
        'MarkerSize',markersize_list(mod(kk-1,5)+1), 'Marker',marker_list{mod(kk-1,5)+1},...
        'MarkerFaceColor',color_style_list{kk},...
        'DisplayName',algo_names{alg});
    hold on;
end
box on
if exist(foldername,'dir')
    set(gca, 'FontSize', 18, 'Xlim', [1,40],...
    'units','pixels','Position',[110 75 650 500]);
else
    set(gca, 'FontSize', 18, 'Xlim', [1,40],...
    'units','pixels','Position',[90 65 570 410]);
end
ylim([3 5000])
xlabel('Frame Number', 'FontSize', 22)
ylabel('nProd by \Phi and \Phi^T', 'FontSize', 22)
% Split legend
leg2 = legend(plot_handle(5:end),'location', 'east');
set(leg2,'box','off');
ah2=axes('units','pixels','position',get(gca,'position'),...
    'FontSize', 18, 'visible','off');
leg2pos = get(leg2,'position');
leg1 = legend(ah2,plot_handle(1:4),'location', 'west');
set(leg1,'box','off');
leg1pos = get(leg1,'position');
set(leg1,'position',[leg2pos(1)-leg1pos(3),leg1pos(2:4)]);
