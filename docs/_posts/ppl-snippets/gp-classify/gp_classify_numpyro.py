# Import libraries.
import json
import matplotlib.pyplot as plt
from jax import random, lax
import jax.numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer import SVI, ELBO
from numpyro.infer import MCMC, NUTS, HMC
import numpy as onp
from sklearn.datasets import make_moons
from tqdm import trange

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

# Default to double precision.
numpyro.enable_x64()


### Make data ###
X, y = make_moons(n_samples=50, shuffle=True, noise=0.1, random_state=1)


### Define Model ###
def sq_exp_cov(X, alpha, rho):
    D = np.linalg.norm(X[:, None] - X, ord=2, axis=-1)
    return alpha * alpha * np.exp(-0.5 * np.power(D / rho, 2))

def compute_f(alpha, rho, beta, eta, X, jitter=1e-6):
    N = X.shape[0]
    K = sq_exp_cov(X, alpha, rho) + np.eye(N) * jitter
    L = np.linalg.cholesky(K)
    return np.matmul(L, eta) + beta

# GP Binary Classifier.
def GPC(X, y):
    N = y.shape[0]
    
    # Priors.
    alpha = numpyro.sample('alpha', dist.LogNormal(0, 1))
    rho = numpyro.sample('rho', dist.LogNormal(0, 1))
    beta = numpyro.sample('beta', dist.Normal(0, 1))
    eta = numpyro.sample('eta', dist.Normal(np.zeros(N), 1))

    # Latent function.
    f = compute_f(alpha, rho, beta, eta, X, 1e-3)
   
    # Likelihood.
    numpyro.sample('obs', dist.Bernoulli(logits=f), obs=y)


### HMC ###
# Set random seed for reproducibility.
rng_key = random.PRNGKey(0)
# NOTE: num_leapfrog = trajectory_length / step_size
hmc = MCMC(HMC(GPC, step_size=.05, trajectory_length=1,
               adapt_step_size=False, adapt_mass_matrix=False),
           num_samples=500, num_warmup=500)
hmc.run(rng_key, X, y)  # Run sampler
hmc_samples = hmc.get_samples()  # Store samples.


### NUTS ###
# Set random seed for reproducibility.
rng_key = random.PRNGKey(0)
nuts = MCMC(NUTS(GPC, target_accept_prob=0.8, max_tree_depth=10),
            num_samples=500, num_warmup=500)
nuts.run(rng_key, X, y)  # run sampler.
nuts_samples = nuts.get_samples() ## collect samplers.


### ADVI ###
# Learn more about ADVI in Numpyro here: 
#   http://num.pyro.ai/en/stable/svi.html

# Compile
guide = AutoDiagonalNormal(GPC)
optimizer = numpyro.optim.Adam(step_size=0.01)
svi = SVI(GPC, guide, optimizer, loss=ELBO())
init_state = svi.init(random.PRNGKey(1), X, y)

# Run optimizer for 1000 iteratons.
state, losses = lax.scan(lambda state, i: 
                         svi.update(state, X, y),
                         init_state, np.arange(1000))

# Extract surrogate posterior.
params = svi.get_params(state)
plt.plot(losses);

def sample_posterior(guide, params, nsamples, seed=1):
    samples = guide.get_posterior(params).sample(
        random.PRNGKey(seed), (nsamples, ))
    # NOTE: Samples are arranged in alphabetical order.
    #       Not in the order in which they appear in the
    #       model. This is different from pyro.
    return dict(alpha=onp.exp(samples[:, 0]),
                beta=samples[:, 1],
                eta=samples[:, 2:52],
                rho=onp.exp(samples[:, -1]))

advi_samples = sample_posterior(guide, params,
                                nsamples=500, seed=1)


# NOTE: See notebook to see full example.
