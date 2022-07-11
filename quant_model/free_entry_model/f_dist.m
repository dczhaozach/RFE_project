function output = f_dist(xval, config, c_e)

%% load parameters and results

%% Internally calibrated parameters
c_reg = xval.c_reg;                                        % fixed costs of e firms
c_f = xval.c_f;
c_time = xval.c_time;

delta = xval.delta;                                        % exogensous exit rate
rho = xval.rho;                                          % productivity shocks persistence
sig_epsi = xval.sig_epsi;                                     % ... std of productivity shocks
log_s0_mean = xval.log_s0_mean;                                  % distribution of inital productivity
sig_s0 = xval.sig_s0;
sig_z = xval.sig_z;     

%% Parameters
interp_type = config.interp_type;                        
extrap_type = config.extrap_type;

% parameters
beta = config.beta;                                      % labor share
sigma = config.sigma;                                    % discounting factor
eta = config.eta;                                        % labor growth rate

ptile = config.ptile;                                    % minimum p tile for distribution

% persistent productivity
n_z = config.n_z;                                        % number of z grids
log_z_val = - sig_z^2/2;                                % normalize the expected value = 1
ptlie_z = linspace(ptile,1 - ptile,n_z);                    % inv probability
z = norminv(ptlie_z,log_z_val, sig_z)';                 % log z grid

% temporary productivity shocks
n_s = config.n_s;                                        % number of s grids
sig_s = sig_epsi/((1 - rho^2)^0.5);                     % stationay std for s
log_st_val = - sig_s^2/2 ;                              % normalize the expected value = 1
s_mul = config.s_mul;                                    % range parameters for tauchen (1986)
s_max = s_mul * sig_s + log_st_val;                     % max s
s_min = - s_mul * sig_s + log_st_val;                   % min s 
s = linspace(s_min,s_max,n_s)';                         % s grids

% Transition probability for shocks
grid = s;
P = pdist(rho, sig_epsi, grid, log_st_val);           % transition matrix from discretizing AR(1) of s

L_supply = config.L_supply;

% Entry Distribution
en_p_z = disc_npdf(z,log_z_val,sig_z);                      % Initial(and thereafter) z type distribution
en_p_s = disc_npdf(s,log_s0_mean,sig_s0);                   % initial distribution of s: normal

%% output
output = f_equ(xval, config, c_e);

l_policy = output.l_policy;

I_nq_cell = output.I_nq_cell;
dist_en = output.dist_en;

%% Transition matrix of s
Q_s = sparse(kron(P, eye(n_z)));

%% Equilibriunm distribution
% ms: incumbents at start of time t
% ms_op: all firms operating at time t
ms_old = zeros(n_z*n_s,1)./(n_z*n_s);                   % incumbents
ms_err_tol = 1e-14;
ms_err  = 10;
num_itr_ms = 0;
m_en_guess = 1;

itr_ms_tol = 3000;

ms_cell = cell(c_time,1);
ms_cell{1} = m_en_guess .* dist_en;
ms = ms_cell{1};

for t = 2:c_time

ms_cell{t} = ((ms_cell{t-1}' .* ((1 - delta)/(1 + eta)) .* I_nq_cell{t-1}')* Q_s)';
ms = ms + ms_cell{t}; 
end

while ms_err > ms_err_tol && num_itr_ms < itr_ms_tol
    num_itr_ms = num_itr_ms + 1;
    new_ms_old= ( (ms_old' .* ((1 - delta)/(1 + eta)) .* I_nq_cell{c_time}')* Q_s ...  % incumbents
        + ms_cell{c_time}' )';
    ms_err = max(abs(new_ms_old - ms_old),[],'all');
    ms_old = new_ms_old;
end

% calculate distribution

ms = ms + ms_old;
L_demand_total = ms' * l_policy;
m_en = m_en_guess * L_supply / L_demand_total;

%% repeat
ms_old = zeros(n_z*n_s,1)./(n_z*n_s);                   % incumbents
ms_err  = 10;
num_itr_ms = 0;

ms_cell{1} = m_en .* dist_en;
ms = ms_cell{1};
ms_quit = cell(c_time, 1);
ms_quit_all = 0;
for t = 2:c_time
ms_cell{t} = ((ms_cell{t-1}' .* ((1 - delta)/(1 + eta)) .* I_nq_cell{t-1}')* Q_s)';
ms_quit{t-1} = (ms_cell{t-1}' * (1 - ((1 - delta)) .* I_nq_cell{t-1}))./(1 + eta);
ms_quit_all = ms_quit_all + ms_quit{t-1};
ms = ms + ms_cell{t}; 
end

while ms_err > ms_err_tol && num_itr_ms < itr_ms_tol
    num_itr_ms = num_itr_ms + 1;
    new_ms_old = ( (ms_old' .* ((1 - delta)/(1 + eta)) .* I_nq_cell{c_time}')* Q_s ...  % incumbents
        + ms_cell{c_time}' )';
    ms_err = max(abs(new_ms_old - ms_old),[],'all');
    ms_old = new_ms_old;
end

ms = ms + ms_old;

m_all = sum(ms);
m_ex = ms_quit_all + (ms_old' * (1 - I_nq_cell{c_time} .* (1 - delta)) )./(1 + eta);

m_ex_1 = m_en - m_en .* (dist_en' * I_nq_cell{1});

exit_rate = m_ex ./ m_all;
exit_rate_1 = m_ex_1 ./ m_en;
entry_rate = m_en ./ m_all;
L_demand_total = ms' * l_policy;

%% Output
output.m_en = m_en;
output.m_ex = m_ex;
output.m_all = m_all;
output.ms = ms;
output.exit_rate = exit_rate;
output.entry_rate = entry_rate;
output.exit_rate_1 = exit_rate_1;
output.L_demand_total = L_demand_total;

end