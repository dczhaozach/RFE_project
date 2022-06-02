function output = f_dist(xval, config)

%% load parameters and results

%% Internally calibrated parameters
c_reg = xval.c_reg;                                        % fixed costs of e firms
c_f = xval.c_f;
c_e = xval.c_e;                                         % entry costs

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
output = f_equ(xval, config);
l_policy = output.l_policy;
I_nq = output.I_nq;
I_nq_one = output.I_nq_one;
dist_en = output.dist_en;

%% Transition matrix of s
Q_s = sparse(kron(P, eye(n_z)));

%% Equilibriunm distribution
% ms: incumbents at start of time t
% ms_op: all firms operating at time t
ms = zeros(n_z*n_s,1)./(n_z*n_s);                   % incumbents
ms_err_tol = 1e-14;
ms_err  = 10;
num_itr_ms = 0;
m_en_guess = 1;

itr_ms_tol = 3000;

while ms_err > ms_err_tol && num_itr_ms < itr_ms_tol
    num_itr_ms = num_itr_ms + 1;
    new_ms= ( (ms' .* ((1 - delta)/(1 + eta)) .* I_nq_one')* Q_s ...  % incumbents
        + (m_en_guess .* dist_en') )';
    ms_err = max(abs(new_ms - ms),[],'all');
    ms = new_ms;
end

% calculate distribution
L_demand_total = ms' * l_policy;
m_en = m_en_guess * L_supply / L_demand_total;

ms_err  = 10;
num_itr_ms = 0;
while ms_err > ms_err_tol && num_itr_ms < itr_ms_tol
    num_itr_ms = num_itr_ms + 1;
    new_ms= ( (ms' .* ((1 - delta)/(1 + eta)) .* I_nq_one')* Q_s ...  % incumbents
        + (m_en .* dist_en') )';
    ms_err = max(abs(new_ms - ms),[],'all');
    ms = new_ms;
end

m_all = sum(ms);
m_ex = (m_all - (ms'  * I_nq_one) .* (1 - delta))./(1 + eta);

m_ex_1 = m_en - m_en .* (dist_en' * I_nq);

exit_rate = m_ex ./ m_all;
exit_rate_1 = m_ex_1 ./ m_en;
entry_rate = m_en ./ m_all;

%% Output
output.m_en = m_en;
output.m_ex = m_ex;
output.m_all = m_all;
output.ms = ms;
output.exit_rate = exit_rate;
output.entry_rate = entry_rate;
output.exit_rate_1 = exit_rate_1;

end