# NOTE: Import libraries ...

# One-dimensional squared exponential kernel with diagonal noise term.
def squared_exp_cov_1D(X, variance, lengthscale):
    deltaXsq = np.power((X[:, None] - X) / lengthscale, 2.0)
    K = variance * np.exp(-0.5 * deltaXsq)
    return K

# GP model.
def GP(X, y):
    # Set informative log-normal priors on kernel hyperparameters.
    variance = numpyro.sample("kernel_var", dist.LogNormal(0.0, 0.1))
    lengthscale = numpyro.sample("kernel_length", dist.LogNormal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 1.0))

    # Compute kernel
    K = squared_exp_cov_1D(X, variance, lengthscale)
    K += np.eye(X.shape[0]) * np.power(sigma, 2)

    # Sample y according to the standard gaussian process formula
    numpyro.sample("y", dist.MultivariateNormal(loc=np.zeros(X.shape[0]),
                                                covariance_matrix=K), obs=y)


# Set random seed for reproducibility.
rng_key = random.PRNGKey(0)


### Fit GP via HMC ###
# NOTE: num_leapfrog = trajectory_length / step_size
kernel = HMC(GP, step_size=.01, trajectory_length=1)
hmc = MCMC(kernel, num_samples=1000, num_warmup=1000)
hmc.run(rng_key, X, y)
hmc_samples = hmc.get_samples()


### Fit GP via NUTS ###
kernel = NUTS(GP, max_tree_depth=10, target_accept_prob=0.8
nnuts = MCMC(kernel, num_samples=1000, num_warmup=1000)
nuts.run(rng_key, X, y)
nuts_samples = hmc.get_samples()


### NOTE: Could not successfully implement ADVI. ###
