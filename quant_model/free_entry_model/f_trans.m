function output = f_trans(xval, config, c_reg_seq, T)

%% load parameters and results

%% Internally calibrated parameters
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

%% output at the beginning and the end
c_e = xval.c_e + sum(c_reg_seq);
output_station_end = f_dist(xval, config, c_e);

EV_end = output_station_end.EV;

c_e = xval.c_e;
output_station_init = f_dist(xval, config, c_e);

dist_en = output_station_init.dist_en;
ms_init= output_station_init.ms;

%% Transition matrix of s
EV = EV_end;
I_nq_seq = zeros(T,n_z*n_s);
l_policy_seq = zeros(T,n_z*n_s);
w_seq = zeros(T,1);

%% transitional path
for k = 1:T
    t = T - k + 1;
    c_reg = c_reg_seq(t);
    c_reg_sum = sum(c_reg_seq(1:t));
    output_t = f_equ_trans(xval, config, c_reg, c_reg_sum, EV);
    EV = output_t.EV;
    I_nq_seq(t,:) = output_t.I_nq;
    l_policy_seq(t,:) = output_t.l_policy;
    w_seq(t) = output_t.w;
end


%%

Q_s = sparse(kron(P, eye(n_z)));
ms = ms_init;
ms_seq = zeros(T, n_z*n_s);
m_all_seq = zeros(T, 1);
m_ex_seq = zeros(T, 1);
exit_rate_seq = zeros(T, 1);
entry_rate_seq = zeros(T, 1);

for t = 1:T
    I_nq = I_nq_seq(t,:)';
    l_policy = l_policy_seq(t,:)';
    
    % calculate distribution
    L_demand_total = ms' * l_policy;
    m_en = (L_supply - L_demand_total)/(dist_en' * l_policy);
    m_en = max(0, m_en);
    
    %% update distribution
    new_ms= ( (ms' .* ((1 - delta)/(1 + eta)) .* I_nq')* Q_s ...  % incumbents
        + (m_en .* dist_en') )';

    ms_seq(t,:) = new_ms;
   
    %% update measures
    m_all = sum(new_ms);
    m_all_seq(t) = m_all;
    m_ex = (sum(ms) - (ms'  * I_nq) .* (1 - delta))./(1 + eta);
    m_ex_seq(t)= m_ex;
    exit_rate_seq(t) = m_ex ./ m_all;
    entry_rate_seq(t) = m_en ./ m_all;
    ms = new_ms;
        
end


%% Output
output.m_all_seq = m_all_seq;
output.ms_seq = ms_seq;
output.exit_rate = exit_rate_seq;
output.entry_rate = entry_rate_seq;

end