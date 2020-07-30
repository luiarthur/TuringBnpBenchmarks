# NOTE: Import library ...

# NOTE: Read Data ...

# Define GP Model
def make_gp_model(X, y, noise=torch.tensor(1e-3).sqrt(),
                  length_prior=dist.LogNormal(-2, 0.1),
                  variance_prior=dist.LogNormal(0.0, 1.0)):
    
    # Define squared exponential covariance function.
    cov_fn = gp.kernels.RBF(input_dim=1)

    # Define GP regression model.
    gpr = gp.models.GPRegression(X, y, cov_fn, noise=noise)

    # Place priors on GP covariance function parameters.
    gpr.kernel.lengthscale = pyro.nn.PyroSample(length_prior)
    gpr.kernel.variance = pyro.nn.PyroSample(variance_prior)
    
    return gpr


### HMC ###
pyro.clear_param_store()  # Clear parameter cache.
pyro.set_rng_seed(1)  # Set random seed for reproducibility.
hmc_gpr = make_gp_model(X, y) # Make GP model for HMC.
# Set up HMC sampler.
kernel = HMC(hmc_gpr.model, step_size=0.01, trajectory_length=1,
             target_accept_prob=0.8, adapt_step_size=False,
             adapt_mass_matrix=False)
hmc = MCMC(kernel, num_samples=1000, warmup_steps=1000)
hmc.run()  # Run sampler.
hmc_posterior_samples = hmc.get_samples() # Get posterior samples


## NUTS ###
pyro.clear_param_store() 
pyro.set_rng_seed(1)
nuts_gpr = make_gp_model(X, y)
kernel = NUTS(nuts_gpr.model, target_accept_prob=0.8)
nuts = MCMC(kernel, num_samples=1000, warmup_steps=1000)
nuts.run()
nuts_posterior_samples = nuts.get_samples()

# NOTE: Could not implement ADVI.
