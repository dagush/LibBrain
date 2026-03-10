%% 1. Configuration and Paths
clear all; close all; clc;

% --- PATHS ---
BASE_DIR = 'L:\Dpt. IMAE Dropbox\Gustavo Patow\SRC\LibBrain\MiniNeuroNumba\_Data_Raw\';
% DIR_DTI = fullfile(BASE_DIR, 'Structural July 2023');
DIR_DATA = fullfile(BASE_DIR, '') % fullfile(BASE_DIR, 'Data July 2023');

% --- USER SETTINGS ---
GROUP = 'CNT';
SUBJECT_ID = 'S01'; 
TR = 2.0;            % Repetition Time (seconds)
Tmax_vol = 295;      % Number of volumes (TRs) to simulate
NUM_RUNS = 10;       % Number of simulations to average

% --- MODEL PARAMETERS ---
G = 5.30;            % Global Coupling
Delta_i = 47.0;      % Inhibitory Heterogeneity
alpha_fic = 3.1415;    % FIC tuning parameter

% Standard Montbrio Parameters (Sanz-Perl)
Jee = 50; 
Jie = 20; 
Jii = -20; 
Jei_base = -20;      
Ie1 = 30; 
Ii1 = 40; 
Deltae = 40;
sigma = 0.01;        
gamma = 1;

%% 2. Load Empirical Data

% A. Load Structural Connectivity (SC)
% sc_folder = fullfile(DIR_DTI, [GROUP '_structure_mat_files']);
% sc_file = fullfile(sc_folder, [GROUP '_' SUBJECT_ID '_structure.mat']);
sc_file = fullfile(BASE_DIR, [GROUP '_' SUBJECT_ID '_structure.mat']);
sc_var_name = [GROUP '_' SUBJECT_ID '_structure'];

fprintf('Loading SC from: %s\n', sc_file);
temp_sc = load(sc_file);
C = temp_sc.(sc_var_name);
C = C / max(max(C)) * 0.2; % Normalize
N = size(C, 1); 

% B. Load Empirical fMRI
fmri_file = fullfile(DIR_DATA, [GROUP '.mat']);
fprintf('Loading fMRI from: %s\n', fmri_file);
temp_fmri = load(fmri_file);
field_names = fieldnames(temp_fmri);
data_3d = temp_fmri.(field_names{1}); 

if ndims(data_3d) == 3
    ts_emp_raw = squeeze(data_3d(1, :, :)); 
else
    ts_emp_raw = data_3d;
end

% Reorder AAL to Deco
left_idx = 1:2:90; 
right_idx = 90:-2:2;
order_deco = [left_idx, right_idx];
ts_emp = ts_emp_raw(order_deco, :);
ts_emp = detrend(ts_emp')';

%% 3. Filters (Bandpass 0.008 - 0.08 Hz)
flp = 0.008; 
fhi = 0.08;
k = 2; 
fnq = 1 / (2 * TR);
Wn = [flp/fnq fhi/fnq];
[bfilt, afilt] = butter(k, Wn);

% Filter Empirical
ts_emp_filt = zeros(size(ts_emp));
for n = 1:N
    ts_emp_filt(n,:) = filtfilt(bfilt, afilt, ts_emp(n,:));
end
FC_emp = corrcoef(ts_emp_filt');

%% 4. Run N Simulation Loops
fprintf('Starting %d runs (G=%.2f, Alpha_FIC=%.2f)...\n', NUM_RUNS, G, alpha_fic);

% Storage for all runs (N x N x Num_Runs)
FC_storage = nan(N, N, NUM_RUNS);

dt = 1e-5; % Default 1e-3, decrease if model explodes. For high G almost mandatory to have super small dt         
T_sim_seconds = (Tmax_vol * TR) + 20; % 20 seconds of warmup
T_steps = round(T_sim_seconds / dt);
ds_step = 10; 
n_store = floor(T_steps / ds_step);

etavec = repmat([Ie1; Ii1], N, 1);
Deltavec = repmat([Deltae; Delta_i], N, 1);
Jglob = Jee;

% --- FIC CALCULATION (Herzog way) ---
Node_Strength = sum(C, 2); 
FIC_Factor = (alpha_fic * G * Node_Strength) + 1;
Jei_vec = Jei_base .* FIC_Factor; 

% Build Coupling Matrix
A_mat = zeros(2*N, 2*N);
for m = 1:N
    idx_e = 1 + 2*(m-1);
    idx_i = 2 + 2*(m-1);
    A_mat(idx_e, idx_e) = Jee;
    A_mat(idx_i, idx_i) = Jii;
    A_mat(idx_e, idx_i) = Jei_vec(m); 
    A_mat(idx_i, idx_e) = Jie;
    for n = 1:N
        idx_e_n = 1 + 2*(n-1);
        A_mat(idx_e, idx_e_n) = A_mat(idx_e, idx_e_n) + G * Jglob * C(m,n);
    end
end

% --- LOOP OVER RUNS ---
for run_idx = 1:NUM_RUNS
    fprintf('  -> Run %d/%d... ', run_idx, NUM_RUNS);
    
    % Reset Initial Conditions
    r = repmat([0.7813; 0.7813], N, 1);
    v = repmat([-0.4196; -0.4196], N, 1);
    rold = r; vold = v;
    
    r_store = zeros(N, n_store);
    store_idx = 1;
    diverged = false;
    
    % Simulation Integration
    for t = 1:T_steps
        kr = (Deltavec/pi + 2.*rold.*vold);
        kv = (vold.^2 + A_mat*rold + etavec - pi^2.*rold.^2);
        
        % Noise ensures each run is different
        noise_r = sqrt(dt*gamma)*sigma*randn(2*N, 1);
        noise_v = sqrt(dt*gamma)*sigma*randn(2*N, 1);
        
        r = rold + (dt*gamma).*kr + noise_r;
        v = vold + (dt*gamma).*kv + noise_v;
        
        % FAILSAFE
        if any(isnan(r)) || any(isinf(r)) || any(r > 500)
            diverged = true;
            break; 
        end
        
        rold = r; vold = v;
        
        if mod(t, ds_step) == 0 && store_idx <= n_store
            r_store(:, store_idx) = r(1:2:end);
            store_idx = store_idx + 1;
        end
    end
    
    if diverged
        fprintf('EXPLODED. Skipping.\n');
        FC_storage(:, :, run_idx) = nan; % Mark as bad
    else
        % Compute BOLD
        dt_bold = dt * ds_step; 
        T_total = n_store * dt_bold;
        bds_raw = zeros(N, n_store);
        
        for n = 1:N
            bds_raw(n, :) = BOLD(T_total, r_store(n, :));
        end

        warmup_pts = round(20 / dt_bold);
        step_tr = round(TR / dt_bold);
        valid_indices = (warmup_pts+1):step_tr:n_store;

        if length(valid_indices) > Tmax_vol
            valid_indices = valid_indices(1:Tmax_vol);
        end
        
        ts_sim = bds_raw(:, valid_indices);
        
        % Filter & FC
        ts_sim_filt = zeros(size(ts_sim));
        for n = 1:N
            sig = detrend(ts_sim(n, :));
            ts_sim_filt(n, :) = filtfilt(bfilt, afilt, sig);
        end
        
        FC_this_run = corrcoef(ts_sim_filt');
        FC_storage(:, :, run_idx) = FC_this_run;
        fprintf('Done.\n');
        
        % Keep the last run's time series for plotting illustration
        if run_idx == NUM_RUNS || ~exist('ts_final_demo', 'var')
             ts_final_demo = ts_sim_filt;
        end
    end
end

%% 5. Average Results
% Compute mean FC ignoring any NaN runs
FC_sim_avg = mean(FC_storage, 3, 'omitnan');

% Compute KS on the Averaged FC
mask = triu(true(N), 1);
[~, ~, ks_stat] = kstest2(FC_emp(mask), FC_sim_avg(mask));
fprintf('\nAll Runs Completed.\n');
fprintf('Average KS Distance: %.4f\n', ks_stat);

%% 6. Plotting
figure('Position', [100, 100, 1200, 500], 'Color', 'w');

subplot(1, 2, 1);
imagesc(FC_emp); axis square; caxis([0 1]); colorbar;
title('Empirical FC');

subplot(1, 2, 2);
if all(isnan(FC_sim_avg(:)))
    text(0.5, 0.5, 'ALL RUNS EXPLODED', 'HorizontalAlignment', 'center', 'Color', 'r');
    axis off;
else
    imagesc(FC_sim_avg); axis square; caxis([0 1]); colorbar;
    title(sprintf('Averaged Simulated FC (%d Runs, G=%.2f)', NUM_RUNS, G));
end

figure('Position', [100, 650, 1200, 600], 'Color', 'w');
subplot(2, 1, 1);
plot(ts_emp_filt', 'Color', [0 0 0 0.1]); hold on;
plot(mean(ts_emp_filt, 1), 'r', 'LineWidth', 2);
title('Empirical BOLD Signals'); xlim([0 size(ts_emp_filt, 2)]);

subplot(2, 1, 2);
if exist('ts_final_demo', 'var')
    plot(ts_final_demo', 'Color', [0 0 1 0.1]); hold on;
    plot(mean(ts_final_demo, 1), 'r', 'LineWidth', 2);
    title(sprintf('Simulated BOLD Signals (Example)', G)); xlim([0 size(ts_final_demo, 2)]);
    xlabel('Time (TRs)');
else
    text(0.5, 0.5, 'NO VALID SIGNAL', 'HorizontalAlignment', 'center', 'Color', 'r');
    axis off;
end

%% ========================================================================
%  LOCAL BOLD FUNCTION (Stephan 2007)
%  =======================================================================
function [b] = BOLD(T,r)
    n_t = length(r);
    dt  = T / n_t; 
    taus   = 0.65; tauf   = 0.41; tauo   = 0.98; alpha  = 0.32;
    itaus  = 1/taus; itauf  = 1/tauf; itauo  = 1/tauo; ialpha = 1/alpha;
    Eo     = 0.4; TE     = 0.04; vo     = 0.04; theta0 = 40.3; r0     = 25;
    k1 = 4.3 * theta0 * Eo * TE; k2 = r0 * Eo * TE; k3 = 1;
    x = zeros(n_t, 4); x(1,:) = [0 1 1 1]; 
    for n = 1:n_t-1
        x(n+1,1) = x(n,1) + dt*( r(n) - itaus*x(n,1) - itauf*(x(n,2)-1) );
        x(n+1,2) = x(n,2) + dt*x(n,1);
        x(n+1,3) = x(n,3) + dt*itauo*( x(n,2) - x(n,3)^ialpha );
        q_term = (x(n,2)*(1-(1-Eo)^(1/x(n,2)))/Eo - (x(n,3)^ialpha)*x(n,4)/x(n,3));
        x(n+1,4) = x(n,4) + dt*itauo*q_term;
    end
    q = x(:,4); v = x(:,3);
    b = vo*( k1.*(1-q) + k2.*(1-q./v) + k3.*(1-v) );
end