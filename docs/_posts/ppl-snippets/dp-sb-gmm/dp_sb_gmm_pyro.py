# Stick break function
# See: https://pyro.ai/examples/dirichlet_process_mixture.html
def stickbreak(v):
    cumprod_one_minus_v = torch.cumprod(1 - v, dim=-1)
    one_v = pad(v, (0, 1), value=1)
    c_one = pad(cumprod_one_minus_v, (1, 0), value=1)
    return one_v * c_one

# See: https://pyro.ai/examples/gmm.html#
# See: https://pyro.ai/examples/dirichlet_process_mixture.html
# See: https://forum.pyro.ai/t/fitting-models-with-nuts-is-slow/1900

# DP SB GMM model.
# NOTE: In pyro, priors are assigned to parameters in the following manner:
#
#   random_variable = pyro.sample('name_of_random_variable', some_distribution)
#
# Note that random variables appear on the left hand side of the `pyro.sample` statement.
# Data will appear *inside* the `pyro.sample` statement, via the obs argument.
# 
# In this example, labels are explicitly mentioned. But they are, in fact, marginalized
# out automatically by pyro. Hence, they do not appear in the posterior samples.
#
# Both marginalized and auxiliary variabled versions are equally slow.
def dp_sb_gmm(y, num_components):
    # Cosntants
    N = y.shape[0]
    K = num_components
    
    # Priors
    # NOTE: In pyro, the Gamma distribution is parameterized with shape and rate.
    # Hence, Gamma(shape, rate) => mean = shape/rate
    alpha = pyro.sample('alpha', dist.Gamma(1, 10))
    
    with pyro.plate('mixture_weights', K - 1):
        v = pyro.sample('v', dist.Beta(1, alpha, K - 1))
    
    eta = stickbreak(v)
    
    with pyro.plate('components', K):
        mu = pyro.sample('mu', dist.Normal(0., 3.))
        sigma = pyro.sample('sigma', dist.Gamma(1, 10))

    with pyro.plate('data', N):
        label = pyro.sample('label', dist.Categorical(eta), infer={"enumerate": "parallel"})
        pyro.sample('obs', dist.Normal(mu[label], sigma[label]), obs=y)

# Fit DP SB GMM via ADVI
# See: https://pyro.ai/examples/dirichlet_process_mixture.html

# Automatically define variational distribution (a mean field guide).
pyro.clear_param_store()  # clear global parameter cache
guide = AutoDiagonalNormal(pyro.poutine.block(dp_sb_gmm, expose=['alpha', 'v', 'mu', 'sigma']))
svi = SVI(dp_sb_gmm, guide, Adam({'lr': 1e-2}), TraceEnum_ELBO())
pyro.set_rng_seed(7)  # set random seed
# Do gradient steps.
for step in range(2000):
    svi.step(y, 10)

# Fit DP SB GMM via HMC
pyro.clear_param_store()
pyro.set_rng_seed(1)
kernel = HMC(dp_sb_gmm, step_size=0.01, trajectory_length=1,
             target_accept_prob=0.8, adapt_step_size=False,
             adapt_mass_matrix=False)
hmc = MCMC(kernel, num_samples=500, warmup_steps=500)
hmc.run(y, 10)

# Fit DP SB GMM via NUTS
pyro.clear_param_store()
pyro.set_rng_seed(1)
kernel = NUTS(dp_sb_gmm, target_accept_prob=0.8)
nuts = MCMC(kernel, num_samples=500, warmup_steps=500)
nuts.run(y, 10)
