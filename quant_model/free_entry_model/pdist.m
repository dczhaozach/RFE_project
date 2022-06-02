function P = pdist(rho, sig_s, grid, ss_val)
N = length(grid);
P = zeros(N,N);

for j=1:N
    P(j,1) = normcdf((grid(1) + (grid(2) - grid(1))/2 - (1-rho)*ss_val - rho*grid(j))/sig_s,0,1);
    
    for k=2:(N-1)
        P(j,k) = normcdf((grid(k) + (grid(k+1) - grid(k))/2 - (1-rho)*ss_val - rho*grid(j))/sig_s,0,1)...
            - normcdf((grid(k) - (grid(k) - grid(k-1))/2- (1-rho)*ss_val - rho*grid(j))/sig_s,0,1);
    end
    P(j,N) = 1-normcdf((grid(N) - (grid(N) - grid(N-1))/2 - (1-rho)*ss_val - rho*grid(j))/sig_s,0,1);
end
end