%% Internally calibrated parameters
load config_file.mat

xval.c_reg = 0.0;                                        % fixed costs of e firms
xval.c_f = 2.2;
c_e = 8.5;                                         % entry costs
xval.c_one = 0;


xval.delta = 0.02;                                        % exogensous exit rate
xval.rho = 0.972;                                          % productivity shocks persistence
xval.sig_epsi = 0.032;                                     % ... std of productivity shocks
xval.log_s0_mean = 0.142;                                  % distribution of inital productivity
xval.sig_s0 = 0.138;
xval.sig_z = 0.078;                                       % distribution of permanent productivity

%%
output = f_dist(xval, config, c_e);
display(output.exit_rate)
display(output.exit_rate_1)
display(output.entry_rate)
display(output.L_demand_total)

%%
T = 100;
c_reg_seq = zeros(T,1);
c_reg_seq(1) = 1;
output_trans = f_trans(xval, config, c_reg_seq, T);

%%
output_trans.exit_rate
output_trans.entry_rate