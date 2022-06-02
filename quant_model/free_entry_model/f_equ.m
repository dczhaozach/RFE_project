function output = f_equ(xval, config)

%% load parameters
c_reg = xval.c_reg;                                        % fixed costs of e firms
c_e = xval.c_e;                                         % entry costs
c_one = xval.c_one;

%% solve for equilibrium wage

% option_f0 = optimoptions('fsolve','Display','iter-detailed',...
%             'TolFun', 1e-5, 'TolX', 1e-5);
%         
option_f0 = optimset('Display','iter','TolX',1e-6);
w_initial = config.w_initial;


% find zeros
EV = @(w) f_VFI(xval, w, config, "EV");
fun = @(w) EV(w) - c_e - c_reg - c_one;
[w_eq,fval] = fzero(fun,w_initial,option_f0);
% warning message
if abs(fval) > 1e-3
    warning('function is away from zero')
end

%% Value and Policy functions
output = f_VFI(xval, w_eq, config, "Equilibrium");

%% Output
output.w = w_eq;

end