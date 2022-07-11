function output = f_VFI(xval, config, w, output_mode)
%%
% This function calculate equilibrium labor demand and distributions of the
%     state variables given state prices and parameters.
% The output contains value functions, policy functions, stationay
%     distribution, transition matrices, and error code.
% mode_VFI:
%   wage: calculate the equlibrium wage
%   dyn: calculate the distribution of the state variables


%% Loading parameters______________________________________________________
%% Internally calibrated parameters
c_f = xval.c_f;
c_reg = xval.c_reg;
delta = xval.delta;                                        % exogensous exit rate
rho = xval.rho;                                          % productivity shocks persistence
sig_epsi = xval.sig_epsi;                                     % ... std of productivity shocks
log_s0_mean = xval.log_s0_mean;                                  % distribution of inital productivity
sig_s0 = xval.sig_s0;
sig_z = xval.sig_z;                                       % distribution of permanent productivity
c_time = xval.c_time;

%% Parameters
interp_type = config.interp_type;                        
extrap_type = config.extrap_type;

% parameters
beta = config.beta;                                      % labor share
sigma = config.sigma;                                    % discounting factor
eta = config.eta;                                        % labor growth rate

ptile = config.ptile;                                    % minimum p tile for distribution

% grids

% persistent productivity
n_z = config.n_z;                                        % number of z grids
log_z_val = - sig_z^2/2;                                % normalize the expected value = 1
ptlie_z = linspace(ptile,1 - ptile,n_z);                    % inv probability
z = norminv(ptlie_z,log_z_val, sig_z)';                 % log z grid
exp_z = exp(z);

% temporary productivity shocks
n_s = config.n_s;                                        % number of s grids
sig_s = sig_epsi/((1 - rho^2)^0.5);                     % stationay std for s
log_st_val = - sig_s^2/2 ;                              % normalize the expected value = 1
s_mul = config.s_mul;                                    % range parameters for tauchen (1986)
s_max = s_mul * sig_s + log_st_val;                     % max s
s_min = - s_mul * sig_s + log_st_val;                   % min s 
s = linspace(s_min,s_max,n_s)';                         % s grids
exp_s = exp(s);

[ZIND, SIND] = ...
    ndgrid((1:1:n_z),(1:1:n_s));              % N-D grids for z, s, a

% Transition probability for shocks
grid = s;
P = pdist(rho, sig_epsi, grid, log_st_val);           % transition matrix from discretizing AR(1) of s



%% 3.VFI___________________________________________________________________
% Initialization
EV = 0;
V = 0;
dif_VFI = 10;                                               % intital difference
dif_tol_VFI = 1e-3;                                         % tolerance of the difference
num_itr = 0;                                                % number of iteration
max_itr = 30;

l_demand = (beta.*exp_z(ZIND(:),1) .* exp_s(SIND(:),1)./w).^(1./(1-beta));
profit = exp_z(ZIND(:),1) .* exp_s(SIND(:),1) .* (l_demand.^beta) ...
    - w .* l_demand - c_f;

while dif_VFI > dif_tol_VFI && num_itr < max_itr
    %% iteration block
    num_itr = num_itr + 1;
    V_next = profit + sigma .* (1 - delta) .* max(0, EV);
          
    
    % firm type selection
    V_next = reshape(V_next,n_z,n_s);    
    dif_VFI = max(abs(V_next - V),[],'all');
    V = V_next;
    
    % intrapolate value function
    V_intra = griddedInterpolant(exp_z(ZIND),exp_s(SIND),V,interp_type,extrap_type);
    
    EV = sum(P(SIND(:),:) .* ...
          V_intra(repmat(exp_z(ZIND(:)),1,n_s),repmat(exp_s',n_z*n_s,1)),2);

%     display(num_itr)
%     display(dif_VFI)

end

%%
V_cell = cell(c_time,1);
EV_cell = cell(c_time,1);
I_nq_cell = cell(c_time,1);

%%
V_cell{c_time} = V(:);
EV_cell{c_time} = EV;
I_nq_cell{c_time} = EV_cell{c_time} > 0;

for t_itr = 1:c_time - 1
    k_itr = c_time - t_itr;
    V_cell{k_itr} =  profit - c_reg + sigma .* (1 - delta) .* max(0, EV_cell{k_itr+1});
    V_temp = reshape(V_cell{k_itr},n_z,n_s);
    V_intra_temp = griddedInterpolant(exp_z(ZIND),exp_s(SIND),V_temp,interp_type,extrap_type);
    EV_cell{k_itr} = sum(P(SIND(:),:) .* ...
          V_intra_temp(repmat(exp_z(ZIND(:)),1,n_s),repmat(exp_s',n_z*n_s,1)),2);
    I_nq_cell{k_itr} = EV_cell{k_itr} > 0;

end

%% entry distribution
% Entry Distribution
en_p_z = disc_npdf(z,log_z_val,sig_z);                      % Initial(and thereafter) z type distribution
en_p_s = disc_npdf(s,log_s0_mean,sig_s0);                   % initial distribution of s: normal
dist_en = sparse(en_p_z(ZIND(:)) .* en_p_s(SIND(:)));

EV_en = dist_en' * V_cell{1};

%% Output
switch output_mode
    case "EV"
        output = EV_en;
        
    case "Equilibrium"        
        output.I_nq_cell = I_nq_cell;
        output.EV_en = EV_en;
        output.EV_cell = EV_cell;
        output.V_cell = V_cell;
        output.l_policy = l_demand;
        output.dist_en = dist_en;
end
        
        
        
        

end
        


