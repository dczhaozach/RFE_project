%% This file contains loading parametes

%% External calibrated parameters
config.beta = 0.9/3*2;                                      % labor share
config.sigma = 0.95;                                         % discounting factor ()
config.dep = 0.10;
config.L_supply = 19.8;
config.ptile = 0.005;
config.w_initial = 1;
config.eta = 0.02;

%% Options
config.interp_type = 'linear';                            % 'makima'
config.extrap_type = 'linear';
config.init_V = 'zero';
config.save_V = 'false';
config.disp_warn = 'false';
config.disp_itr = 'false';                                   % 'all', 'VFI', 'ms'
config.manual_iter_max = 18;
%% Grids
config.s_mul = 5;                                          % range configeter for tauchen (1985)

config.n_z = 10;

config.n_s = 100;


config.n_epi = 100;

%% x initial guess
f_e = 800;                                           % fixed costs of e firms
f_c = 0.98008;                                               % fixed costs of c firms
delta = 0.020799;                                           % exogensous exit rate
rho =  0.984;                                           % productivity shocks persistence
sig_epsilon = 0.0398;                                          % ... std
log_a0_mean = 0.47449;                                          % inital asset distribution
sig_a0 = 0.610;
log_s0_mean = -0.81764;                                         % distribution of inital productivity
sig_s0 = 0.42338;
sig_z = 0.17;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ;                                     % distribution of permanent productivity
lambda = 1.4;
c_e = 0.057641;
w = 1.65;
x_guess = [f_e, f_c, delta, rho, sig_epsilon, ....
        log_a0_mean, sig_a0, log_s0_mean, sig_s0, sig_z, ...
        lambda, c_e, w]';


config.x_guess = x_guess;
save('config_file','config')