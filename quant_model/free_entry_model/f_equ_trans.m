function output = f_equ_trans(xval, config, c_reg, c_reg_sum, EV)

%% load parameters
c_e = xval.c_e;                                         % entry costs

%% solve for equilibrium wage

% option_f0 = optimoptions('fsolve','Display','iter-detailed',...
%             'TolFun', 1e-5, 'TolX', 1e-5);
%         
option_f0 = optimset('Display','iter','TolX',1e-6);
w_initial = config.w_initial;


% find zeros
EV_fun = @(w) f_VFI_trans(xval, config, w, c_reg, EV, "EV");
fun = @(w) EV_fun(w) - c_e - c_reg_sum;
[w_eq,fval] = fzero(fun,w_initial,option_f0);
% warning message
if abs(fval) > 1e-3
    warning('function is away from zero')
end

%% Value and Policy functions
output = f_VFI_trans(xval, config, w_eq, c_reg, EV, "Equilibrium");

%% Output
output.w = w_eq;

end