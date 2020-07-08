# Stick break function
function stickbreak(v)
    K = length(v) + 1
    cumprod_one_minus_v = cumprod(1 .- v)

    eta = [if k == 1 
               v[1]
           elseif k == K
               cumprod_one_minus_v[K-1]
           else
               v[k] * cumprod_one_minus_v[k-1]
           end
           for k in 1:K]

    return eta
end

# DP GMM model under stick-breaking construction
@model dp_gmm_sb(y, K) = begin
    nobs = length(y)

    mu ~ filldist(Normal(0, 3), K)
    sig ~ filldist(Gamma(1, 1/10), K)  # mean = 0.1

    alpha ~ Gamma(1, 1/10)  # mean = 0.1
    v ~ filldist(Beta(1, alpha), K - 1)
    eta = BnpUtil.stickbreak(v)

    log_target = logsumexp(normlogpdf.(mu', sig', y) .+ log.(eta)', dims=2)
    Turing.acclogp!(_varinfo, sum(log_target))
end

# NOTE: 
# Here, y are noisy univariate draws from a mixture distribution with
# 4 components.

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
