function output = ssp_c(n_group, param, phi_input, mode)

eta = param.eta;
eps = param.eps;

d = param.d;
phi = phi_input;
z = param.z;
mu = param.mu;
alpha = param.alpha * ones(1, n_group);

cost = - 1/2 * alpha * log(1 - phi.^2)';


%% equation
syms c w Lh Y
syms y p L [1, n_group]  

init_param = 1 * ones(4+3*n_group, 1);
%%
%hh
equs = [1/c == (Lh^eta) /w, ...
       y == (((1 - d)./(1 - d .* phi)).^eps) .* Y .* (p.^(-eps)), ...
       p == eps/(eps - 1) .* w ./ z, ...
       L == y ./ z, ...
       Lh == L * ((1 - d .* phi) .* mu)', ...
       Y == c + cost, ...
       Y == w * Lh, ...
       ];

vars = [c, w, L, Y, y, p, Lh];

%% solve
results = vpasolve(equs, vars, init_param);


%%
switch mode
    case "optimize"
        try 
            output = double(results.c(1));
        catch
            output = nan;
        end
        
    case "results"
        results.labor_ratio = results.L1/results.L2;
        results.avg_L1 = results.L1/((1 - phi(1)*d(1))*mu(1));
        results.avg_L2 = results.L2/((1 - phi(2)*d(2))*mu(2));
        results.phi = phi;
        output = results;
end
        
    



end