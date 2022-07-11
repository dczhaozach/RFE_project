%% Internally calibrated parameters
load config_file.mat

xval.c_reg = 0.1;                                        % fixed costs of e firms
xval.c_f = 2.2;
c_e = 8.5;                                         % entry costs
xval.c_one = 0;


xval.delta = 0.02;                                        % exogensous exit rate
xval.rho = 0.972;                                          % productivity shocks persistence
xval.sig_epsi = 0.032;                                     % ... std of productivity shocks
xval.log_s0_mean = 0.142;                                  % distribution of inital productivity
xval.sig_s0 = 0.138;
xval.sig_z = 0.078;                                       % distribution of permanent productivity
xval.c_time = 2;

%%
c_reg_seq = (0: 0.5: 10);
c_time_max = 21;
exit_rate_seq = zeros(length(c_reg_seq), c_time_max);

for c_time_itr = 1:c_time_max
    for c_reg_itr = 1:length(c_reg_seq)
        xval.c_reg = c_reg_seq(c_reg_itr);                                        % fixed costs of e firms
        xval.c_time = c_time_itr;
        output = f_dist(xval, config, c_e);
        exit_rate_seq(c_reg_itr, c_time_itr) = output.exit_rate;
        % display(output.exit_rate_1)
        % display(output.entry_rate)
        % display(output.L_demand_total)
    end
end

%%
T = 100;
c_reg_seq = zeros(T,1);
c_reg_seq(1) = 1;
output_trans = f_trans(xval, config, c_reg_seq, T);

%%
output_trans.exit_rate
output_trans.entry_rate