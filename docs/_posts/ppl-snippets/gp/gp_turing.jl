# Import libraries.
using Turing, Turing: Variational
using Distributions
using AbstractGPs, KernelFunctions
import Random
import LinearAlgebra

# NOTE: Import other libraries ...

# NOTE: Read data ...

# Define a kernel.
sqexpkernel(alpha::Real, rho::Real) = 
    alpha^2 * transform(SqExponentialKernel(), 1/(rho*sqrt(2)))

@model function GPRegression(y, X, m_alpha=0.0, s_alpha=1.0, m_rho=0.0,
                             s_rho=1.0, m_sigma=0.0, s_sigma=1.0)
    # Priors.
    alpha ~ LogNormal(m_alpha, s_alpha)
    rho ~ LogNormal(m_rho, s_rho)
    sigma ~ LogNormal(m_sigma, s_sigma)
    
    # Realized covariance function
    kernel = sqexpkernel(alpha, rho)
    K = kernelmatrix(kernel, X, obsdim=1)
    
    # Sampling Distribution.
    y ~ MvNormal(K + LinearAlgebra.I * sigma^2)  # mean=0, covariance=K + σ²I.
end;

# Set random number generator seed.
Random.seed!(0)  

# Model creation.
m = GP(y, X, sqexp_cov_fn, 0.0, 0.1)


# Fit via ADVI.
q0 = Variational.meanfield(m)  # initialize variational distribution (optional)
num_elbo_samples, max_iters = (1, 2000)
# Run optimizer.
@time q = vi(m, ADVI(num_elbo_samples, max_iters), q0,
             optimizer=Flux.ADAM(1e-1));


# Fit via HMC.
burn = 1000
nsamples = 1000
@time chain = sample(m, HMC(0.01, 100), burn + nsamples)  # start sampling.


# Fit via NUTS.
@time chain = begin
    nsamples = 1000  # number of MCMC samples
    nadapt = 1000  # number of iterations to adapt tuning parameters in NUTS
    iterations = nsamples + nadapt
    target_accept_ratio = 0.8
    
    sample(m, NUTS(nadapt, target_accept_ratio, max_depth=10), iterations);
end
