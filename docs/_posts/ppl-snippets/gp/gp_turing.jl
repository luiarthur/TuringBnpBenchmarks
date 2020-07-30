# NOTE: Import libraries ...

# NOTE: Read data ...

# Squared-exponential covariance function.
function sqexp_cov_fn(D, alpha, rho, eps=1e-3)
  return alpha ^ 2 * exp.(-0.5 * (D/rho) .^ 2) + LinearAlgebra.I * eps
end

# Define GP model.
@model function GP(y, X, cov_fn=sqexp_cov_fn, m_alpha=0.0, s_alpha=1.0,
                   m_rho=0.0, s_rho=1.0)
    # Distance matrix.
    D = pairwise(Distances.Euclidean(), X, dims=1)
    
    # Priors.
    alpha ~ LogNormal(m_alpha, s_alpha)
    rho ~ LogNormal(m_rho, s_rho)
    
    # Realized covariance function
    K = cov_fn(D, alpha, rho)
    
    # Sampling Distribution.
    y ~ MvNormal(K)  # mean=0, covariance=K.
end

# Set random number generator seed.
Random.seed!(0)  

# Fit via ADVI.
m = GP(y, X, sqexp_cov_fn, 0.0, 1.0, -2.0, 0.1)  # Model creation.
q0 = Variational.meanfield(m)  # initialize variational distribution (optional)
num_elbo_samples, max_iters = (1, 2000)
# Run optimizer.
@time q = vi(m, ADVI(num_elbo_samples, max_iters), q0,
             optimizer=Flux.ADAM(1e-1));


# Fit via HMC.
burn = 1000
nsamples = 1000
@time chain = sample(m, HMC(0.01, 100), burn + nsamples)  # start sampling.


# Get posterior samples
alpha = vec(group(chain, :alpha).value.data[end-nsamples:end, :, 1]);
rho = vec(group(chain, :rho).value.data[end-nsamples:end, :, 1]);
hmc_samples = Dict(:alpha => alpha, :rho => rho)

# Fit via NUTS.
@time chain = begin
    nsamples = 1000  # number of MCMC samples
    nadapt = 1000  # number of iterations to adapt tuning parameters in NUTS
    iterations = nsamples + nadapt
    target_accept_ratio = 0.8
    
    sample(m, NUTS(nadapt, target_accept_ratio, max_depth=10), iterations);
end

# Get posterior samples
alpha = vec(group(chain, :alpha).value.data[:, :, 1]);
rho = vec(group(chain, :rho).value.data[:, :, 1]);
nuts_samples = Dict(:alpha => alpha, :rho => rho);
