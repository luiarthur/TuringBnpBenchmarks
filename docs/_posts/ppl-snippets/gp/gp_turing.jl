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

# Define model.
@model GPRegression(y, X) = begin
    # Priors.
    alpha ~ LogNormal(0.0, 0.1)
    rho ~ LogNormal(0.0, 1.0)
    sigma ~ LogNormal(0.0, 1.0)
    
    # Covariance function.
    kernel = sqexpkernel(alpha, rho)

    # Finite GP.
    gp = GP(kernel)
    
    # Sampling Distribution (MvNormal likelihood).
    y ~ gp(X, sigma^2 + 1e-6)  # add 1e-6 for numerical stability.
end;


# Set random number generator seed.
Random.seed!(0)  

# Model creation.
m = GPRegression(y, X)


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
