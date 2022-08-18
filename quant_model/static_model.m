mu_test = 0.5;
param.mu = [mu_test, 1 - mu_test];
%%
d_test = 0.2;
n_group = 2;
phi_guess = [0.2, 0.2];

param.eta = 2;
param.eps = 5;
param.d = [d_test, d_test];
param.z = [0.3, 1];

param.alpha = 0.02;

%% 
grid = (0.1:0.2:1);
n_grid = length(grid);
y_L1 = zeros(n_grid,1);
y_L2 = zeros(n_grid,1);
y_phi1 = zeros(n_grid,1);
y_phi2 = zeros(n_grid,1);
y_labor_ratio = zeros(n_grid,1);
y_Lh = zeros(n_grid,1);
for i = 1:n_grid
    param.z = [grid(i), 1];
    results = gov_policy(n_group, param, phi_guess);
    y_L1(i) = results.L1;
    y_L2(i) = results.L2;
    y_phi1(i)= results.phi(1);
    y_phi2(i)= results.phi(2);
    y_labor_ratio(i) = results.labor_ratio;
    y_Lh(i) = results.Lh;
end

%%
plot(grid, y_phi1)
