# import some libraries here...

# Stick break function
def stickbreak(v):
    batch_ndims = len(v.shape) - 1
    cumprod_one_minus_v = np.exp(np.log1p(-v).cumsum(-1))
    one_v = np.pad(v, [[0, 0]] * batch_ndims + [[0, 1]], constant_values=1)
    c_one = np.pad(cumprod_one_minus_v, [[0, 0]] * batch_ndims +[[1, 0]],
                   constant_values=1)
    return one_v * c_one

# Custom distribution: Mixture of normals.
# This implements the abstract class `dist.Distribution`.
class NormalMixture(dist.Distribution):
    support = constraints.real_vector

    def __init__(self, mu, sigma, w):
        super(NormalMixture, self).__init__(event_shape=(1, ))
        self.mu = mu
        self.sigma = sigma
        self.w = w

    def sample(self, key, sample_shape=()):
        # it is enough to return an arbitrary sample with correct shape
        return np.zeros(sample_shape + self.event_shape)

    def log_prob(self, y, axis=-1):
        lp = dist.Normal(self.mu, self.sigma).log_prob(y) + np.log(self.w)
        return logsumexp(lp, axis=axis)

# DP SB GMM model.
# NOTE: In numpyro, priors are assigned to parameters in the following manner:
#
#   random_variable = numpyro.sample('name_of_random_variable', some_distribution)
#
# Note that random variables appear on the left hand side of the
# `numpyro.sample` statement.  Data will appear *inside* the `numpyro.sample`
# statement, via the obs argument.
# 
# In this example, labels are explicitly mentioned. But they are, in fact,
# marginalized out automatically by numpyro. Hence, they do not appear in the
# posterior samples.
def dp_sb_gmm(y, num_components):
    # Cosntants
    N = y.shape[0]
    K = num_components
    
    # Priors
    # NOTE: In numpyro, the Gamma distribution is parameterized with shape and
    # rate.  Hence, Gamma(shape, rate) => mean = shape/rate
    alpha = numpyro.sample('alpha', dist.Gamma(1, 10))
    
    with numpyro.plate('mixture_weights', K - 1):
        v = numpyro.sample('v', dist.Beta(1, alpha, K - 1))
    
    eta = stickbreak(v)
    
    with numpyro.plate('components', K):
        mu = numpyro.sample('mu', dist.Normal(0., 3.))
        sigma = numpyro.sample('sigma', dist.Gamma(1, 10))

    with numpyro.plate('data', N):
        numpyro.sample('obs', NormalMixture(mu[None, :] , sigma[None, :],
                                            eta[None, :]), obs=y[:, None])

# NOTE: Read data y here ...
# Here, y (a vector of length 200) is noisy univariate draws from a
# mixture distribution with 4 components.

# Set random seed for reproducibility.
rng_key = random.PRNGKey(0)

# FIT DP SB GMM via HMC
# NOTE: num_leapfrog = trajectory_length / step_size
kernel = HMC(dp_sb_gmm, step_size=.01, trajectory_length=1)
hmc = MCMC(kernel, num_samples=500, num_warmup=500)
hmc.run(rng_key, y, 10)
hmc_samples = get_posterior_samples(hmc)

# FIT DP SB GMM via NUTS
kernel = NUTS(dp_sb_gmm, max_tree_depth=10, target_accept_prob=0.8)
nuts = MCMC(kernel, num_samples=500, num_warmup=500)
nuts.run(rng_key, y, 10)
nuts_samples = get_posterior_samples(nuts)

# FIT DP SB GMM via ADVI
sigmoid = lambda x: 1 / (1 + onp.exp(-x))

# Setup ADVI.
guide = AutoDiagonalNormal(dp_sb_gmm)  # surrogate posterior
optimizer = numpyro.optim.Adam(step_size=0.01)  # adam optimizer
svi = SVI(guide.model, guide, optimizer, loss=ELBO())  # ELBO loss
init_state = svi.init(random.PRNGKey(2), y, 10)  # initial state

# Run optimizer
state, losses = lax.scan(lambda state, i: 
                         svi.update(state, y, 10), init_state, np.arange(2000))

# Extract surrogate posterior.
params = svi.get_params(state)
def sample_advi_posterior(guide, params, nsamples, seed=1):
    samples = guide.get_posterior(params).sample(random.PRNGKey(seed),
                                                 (nsamples, ))
    # NOTE: Samples are arranged in alphabetical order.
    #       Not in the order in which they appear in the
    #       model. This is different from pyro.
    return dict(alpha=onp.exp(samples[:, 0]),
                mu=onp.array(samples[:, 1:11]).T,
                sigma=onp.exp(samples[:, 11:21]).T,
                eta=onp.array(stickbreak(sigmoid(samples[:, 21:]))).T)  # v

advi_samples = sample_advi_posterior(guide, params, nsamples=500, seed=1)
