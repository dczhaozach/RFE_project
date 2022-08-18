function results = gov_policy(n_group, param, phi_guess)
%% parameters

%% equation
option_fc = optimoptions('fmincon','Display','iter-detailed',...
    'UseParallel',true, 'PlotFcn', 'optimplotfval');

limits =   [...
    0.001,         1;        ... f_e
    0.001,         1;       ... f_c
    ];

lb = limits(:,1);
ub = limits(:,2);
x0 = phi_guess;
A = [];
b = [];
Aeq = [];
beq = [];
noncon = [];

%%
fun = @(x) - ssp_c(n_group, param, x, "optimize");
[xval,~] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,noncon,option_fc);

%% results
results = ssp_c(n_group, param, xval, "results");
end