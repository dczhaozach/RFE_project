%% Internally calibrated parameters
load config_file.mat

xval.c_reg = 0.0;                                        % fixed costs of e firms
xval.c_f = 2.2;
xval.c_e = 7.5;                                         % entry costs
xval.c_one = 1;

xval.delta = 0.02;                                        % exogensous exit rate
xval.rho = 0.972;                                          % productivity shocks persistence
xval.sig_epsi = 0.032;                                     % ... std of productivity shocks
xval.log_s0_mean = 0.142;                                  % distribution of inital productivity
xval.sig_s0 = 0.138;
xval.sig_z = 0.078;                                       % distribution of permanent productivity

%%
output = f_dist(xval, config);
display(output.exit_rate)
display(output.exit_rate_1)
display(output.entry_rate)