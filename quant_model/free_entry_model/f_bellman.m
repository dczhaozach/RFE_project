function V = f_bellman(xval, config, profit, EV)
sigma = config.sigma;
delta = xval.delta;

V = profit + sigma .* (1 - delta) .* max(0, EV);

end