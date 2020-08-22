# NOTE: Import library ...

# NOTE: Read Data ...

# Define GP Model
def make_gp_model(X, y,
                  length_prior=dist.LogNormal(0.0, 1.0),
                  variance_prior=dist.LogNormal(0.0, 0.1),
                  noise_prior=dist.LogNormal(0.0, 1.0)):
    
    # Define squared exponential covariance function.
    cov_fn = gp.kernels.RBF(input_dim=1)

    # Define GP regression model.
    gpr = gp.models.GPRegression(X, y, cov_fn)

    # Place priors on GP covariance function parameters.
    gpr.kernel.lengthscale = pyro.nn.PyroSample(length_prior)
    gpr.kernel.variance = pyro.nn.PyroSample(variance_prior)
    gpr.noise = pyro.nn.PyroSample(noise_prior)
    
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


### NUTS ###
pyro.clear_param_store() 
pyro.set_rng_seed(1)
nuts_gpr = make_gp_model(X, y)
kernel = NUTS(nuts_gpr.model, target_accept_prob=0.8)
nuts = MCMC(kernel, num_samples=1000, warmup_steps=1000)
nuts.run()
nuts_posterior_samples = nuts.get_samples()


### ADVI ###
pyro.clear_param_store()  # clear global parameter cache
pyro.set_rng_seed(1)  # set random seed

# Automatically define variational distribution (a mean field guide).
guide = AutoDiagonalNormal(gp_model)

# Create SVI object for optimization.
svi = SVI(gp_model, guide, Adam({'lr': 1e-2}), JitTrace_ELBO())

# Do 1000 gradient steps.
advi_loss = []
for step in trange(1000):
    advi_loss.append(svi.step(X, y.double()))
    
# Bijector for advi samples.
def biject(samples):
    return dict(alpha=samples[:, 0].exp().numpy(),
                rho=samples[:, 1].exp().numpy(),
                sigma=samples[:, 2].exp().numpy())

# Get ADVI samples in constrained space.
advi_posterior_samples = biject(guide.get_posterior().sample((1000, )))
