function p = disc_npdf(grid,mu,sigma)
% disc_npdf.m creates approximate normal pdf grids for variables
%   min: minimum of the variable
%   max: maximum of the variable
%   n: total number of grids
    n = length(grid);
    p = zeros(n,1);
    p(1) = normcdf(grid(1) + (grid(2) - grid(1))/2 , mu, sigma);
    for i = 2:n-1
        p(i) = normcdf(grid(i) + (grid(i+1) - grid(i))/2, mu, sigma) ...
             - normcdf(grid(i) - (grid(i) - grid(i-1))/2, mu, sigma);
    end
    p(n) = 1 - normcdf(grid(n) - (grid(n) - grid(n-1))/2, mu, sigma);
end