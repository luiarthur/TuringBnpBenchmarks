# NOTE: Import libraries here ...

# DP GMM model under stick-breaking construction
@model dp_gmm_sb(y, K) = begin
    nobs = length(y)

    mu ~ filldist(Normal(0, 3), K)
    sig ~ filldist(Gamma(1, 1/10), K)  # mean = 0.1

    alpha ~ Gamma(1, 1/10)  # mean = 0.1
    crm = DirichletProcess(alpha)
    v ~ filldist(StickBreakingProcess(crm), K - 1)
    eta = stickbreak(v)

    y .~ UnivariateGMM(mu, sig, Categorical(eta))
end

# NOTE: Read data y here ...
# Here, y (a vector of length 500) is noisy univariate draws from a
# mixture distribution with 4 components.

# Fit DP-SB-GMM with ADVI
advi = ADVI(1, 2000)  # num_elbo_samples, max_iters
q = vi(dp_gmm_sb(y, 10), advi, optimizer=Flux.ADAM(1e-2));

# Fit DP-SB-GMM with HMC
hmc_chain = sample(dp_gmm_sb(y, 500),  # data, number of mixture components
                   HMC(0.01, 100),  # stepsize, number of leapfrog steps
                   1000)  # iterations

# Fit DP-SB-GMM with NUTS
@time nuts_chain = begin
    n_samples = 500  # number of MCMC samples
    nadapt = 500  # number of iterations to adapt tuning parameters in NUTS
    iterations = n_samples + nadapt
    target_accept_ratio = 0.8
    
    sample(dp_gmm_sb(y, 10),  # data, number of mixture components.
           NUTS(nadapt, target_accept_ratio, max_depth=10),
           iterations);
end
