#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('echo "Last updated: `date`"')


# In[5]:


import json
import matplotlib.pyplot as plt
from jax import random, lax
import jax.numpy as np
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer import SVI, ELBO
from numpyro.infer import MCMC, NUTS, HMC
import numpy as onp


# In[19]:


def plot_param_post(params, param_name, param_full_name, figsize=(12, 4),
                    truth=None, plot_trace=True):
    if plot_trace:
        plt.figure(figsize=figsize)
        
    param = params[param_name]

    if plot_trace:
        plt.subplot(1, 2, 1)
        
    plt.boxplot(param.T, whis=[2.5, 97.5], showmeans=True, showfliers=False)
    plt.xlabel('mixture components')
    plt.ylabel(param_full_name)
    plt.title('95% Credible Intervals for {}'.format(param_full_name))
    if truth is not None:
        for line in truth:
            plt.axhline(line, ls=':')

    if plot_trace:
        plt.subplot(1, 2, 2)
        plt.plot(param);
        plt.xlabel('iterations')
        plt.ylabel(param_full_name)
        plt.title('Trace plot of {}'.format(param_full_name));

def plot_all_params(samples):
    # TODO: How to get log-likelihood?
    plot_param_post(samples, 'eta', 'mixture weights', truth=simdata['w'])
    plot_param_post(samples, 'mu', 'mixture means', truth=simdata['mu'])
    plot_param_post(samples, 'sigma', 'mixture scales', truth=simdata['sig'])
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(samples['alpha'], bins=30, density=True);
    plt.xlabel("alpha")
    plt.ylabel("density")
    plt.title("Posterior distribution of alpha");


# In[7]:


# Stick break function
def stickbreak(v):
    batch_ndims = len(v.shape) - 1
    cumprod_one_minus_v = np.exp(np.log1p(-v).cumsum(-1))
    # cumprod_one_minus_v = np.cumprod(1-v, axis=-1)
    one_v = np.pad(v, [[0, 0]] * batch_ndims + [[0, 1]], constant_values=1)
    c_one = np.pad(cumprod_one_minus_v, [[0, 0]] * batch_ndims +[[1, 0]], constant_values=1)
    return one_v * c_one

# x = onp.random.randn(3, 2, 5)
# stickbreak(x);  # last dimension sums to 1.


# In[8]:


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


# In[9]:


# Example:
# rng_key = random.PRNGKey(0)
# mu = dist.Normal(0, 1).sample(rng_key, (3, ))
# sig = dist.Uniform(0, 1).sample(rng_key, (3, ))
# w = np.array([.5, .3, .2])
# x = dist.Normal(0, 1).sample(rng_key, (10, 1))
# lp_x = NormalMixture(mu[None, :], sig[None, :], w[None, :]).log_prob(x[:, None])
# print(lp_x.shape)
# print(lp_x)


# In[10]:


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
        numpyro.sample('obs', NormalMixture(mu[None, :] , sigma[None, :], eta[None, :]),
                       obs=y[:, None])
        # Local variables version:
        # label = numpyro.sample('label', dist.Categorical(eta))
        # numpyro.sample('obs', dist.Normal(mu[label], sigma[label]), obs=y)


# In[11]:


# Read simulated data.
path_to_data = '../../data/sim-data/gmm-data-n200.json'
with open(path_to_data) as f:
  simdata = json.load(f)


# In[12]:


# Convert data to torch.tensor.
y = np.array(simdata['y'])


# ## ADVI

# In[13]:


sigmoid = lambda x: 1 / (1 + onp.exp(-x))


# In[14]:


def sample_advi_posterior(guide, params, nsamples, seed=1):
    samples = guide.get_posterior(params).sample(random.PRNGKey(seed), (nsamples, ))
    # NOTE: Samples are arranged in alphabetical order.
    #       Not in the order in which they appear in the
    #       model. This is different from pyro.
    return dict(alpha=onp.exp(samples[:, 0]),
                mu=onp.array(samples[:, 1:11]).T,
                sigma=onp.exp(samples[:, 11:21]).T,
                eta=onp.array(stickbreak(sigmoid(samples[:, 21:]))).T)  # v


# In[15]:


get_ipython().run_cell_magic('time', '', '\n# Compile\nguide = AutoDiagonalNormal(dp_sb_gmm)\noptimizer = numpyro.optim.Adam(step_size=0.01)\nsvi = SVI(guide.model, guide, optimizer, loss=ELBO())\ninit_state = svi.init(random.PRNGKey(2), y, 10)')


# In[16]:


get_ipython().run_cell_magic('time', '', '\n# Run optimizer\nstate, losses = lax.scan(lambda state, i: \n                         svi.update(state, y, 10), init_state, np.arange(2000))\n\n# Extract surrogate posterior.\nparams = svi.get_params(state)\nplt.plot(losses);\nplt.title("Negative ELBO")\nadvi_samples = sample_advi_posterior(guide, params, nsamples=500, seed=1)')


# In[25]:


plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plot_param_post(advi_samples, 'eta', 'mixture weights', truth=simdata['w'], plot_trace=False)
plt.subplot(2, 2, 2)
plt.hist(advi_samples['alpha'], bins=30, density=True); plt.title("Histogram of alpha")
plt.subplot(2, 2, 3)
plot_param_post(advi_samples, 'mu', 'mixture means', truth=simdata['mu'], plot_trace=False)
plt.subplot(2, 2, 4)
plot_param_post(advi_samples, 'sigma', 'mixture scales', truth=simdata['sig'], plot_trace=False)

plt.tight_layout()


# ## HMC

# In[10]:


def get_posterior_samples(mcmc):
    # Get mu, sigma, v, alpha.
    posterior_samples = mcmc.get_samples()
    
    # Transform v to eta.
    posterior_samples['eta'] = stickbreak(posterior_samples['v'])
    
    return posterior_samples


# In[14]:


get_ipython().run_cell_magic('time', '', '\n# Set random seed for reproducibility.\nrng_key = random.PRNGKey(0)\n\n# NOTE: num_leapfrog = trajectory_length / step_size\nkernel = HMC(dp_sb_gmm, step_size=.01, trajectory_length=1,\n             adapt_step_size=False, adapt_mass_matrix=False)\n\nhmc = MCMC(kernel, num_samples=500, num_warmup=500)\nhmc.run(rng_key, y, 10)\n\nhmc_samples = get_posterior_samples(hmc)')


# In[18]:


plot_all_params(hmc_samples)


# ## NUTS

# In[15]:


get_ipython().run_cell_magic('time', '', '\n# Set random seed for reproducibility.\nrng_key = random.PRNGKey(0)\n\n# Set up NUTS sampler.\nkernel = NUTS(dp_sb_gmm, max_tree_depth=10, target_accept_prob=0.8)\n\nnuts = MCMC(kernel, num_samples=500, num_warmup=500)\nnuts.run(rng_key, y, 10)\n\nnuts_samples = get_posterior_samples(nuts)')


# In[19]:


plot_all_params(nuts_samples)


# In[ ]:




