# import some libraries here...

# Stick break function
def stickbreak(v):
    cumprod_one_minus_v = np.exp(np.log1p(-v).cumsum(-1))
    one_v = np.pad(v, (0, 1), constant_values=1)
    c_one = np.pad(cumprod_one_minus_v, (1, 0), constant_values=1)
    return one_v * c_one

# Log sum exp
def logsumexp(x, axis=0, keepdims=False):
    mx = np.max(x, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(x - mx), axis=axis, keepdims=True)) + mx
    if keepdims:
        return out
    else:
        return out.sum(axis=axis)

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
# Note that random variables appear on the left hand side of the `numpyro.sample` statement.
# Data will appear *inside* the `numpyro.sample` statement, via the obs argument.
# 
# In this example, labels are explicitly mentioned. But they are, in fact, marginalized
# out automatically by numpyro. Hence, they do not appear in the posterior samples.
def dp_sb_gmm(y, num_components):
    # Cosntants
    N = y.shape[0]
    K = num_components
    
    # Priors
    # NOTE: In numpyro, the Gamma distribution is parameterized with shape and rate.
    # Hence, Gamma(shape, rate) => mean = shape/rate
    alpha = numpyro.sample('alpha', dist.Gamma(1, 10))
    
    with numpyro.plate('mixture_weights', K - 1):
        v = numpyro.sample('v', dist.Beta(1, alpha, K - 1))
    
    eta = stickbreak(v)
    
    with numpyro.plate('components', K):
        mu = numpyro.sample('mu', dist.Normal(0., 3.))
        sigma = numpyro.sample('sigma', dist.Gamma(1, 10))

    with numpyro.plate('data', N):
        numpyro.sample('obs', NormalMixture(mu[None, :] , sigma[None, :], eta[None, :]), obs=y[:, None])

# NOTE: Read data y here ...
# Here, y (a vector of length 500) is noisy univariate draws from a
# mixture distribution with 4 components.

# NOTE: Due to lack of documentation, I was not able to implement
# an example via ADVI in numpyro.

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
