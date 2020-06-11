#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
print('Last updated: ', datetime.datetime.now(), '(PT)')


# In[2]:


import json
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
# from pyro.optim import Adam
# from pyro.infer import SVI, Trace_ELBO
# from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
# from pyro.contrib.autoguide import AutoDiagonalNormal
from numpyro.infer import MCMC, NUTS, HMC


# In[3]:


# Stick break function
def stickbreak(v):
    cumprod_one_minus_v = np.exp(np.log1p(-v).cumsum(-1))
    # cumprod_one_minus_v = np.cumprod(1-v, axis=-1)
    one_v = np.pad(v, (0, 1), constant_values=1)
    c_one = np.pad(cumprod_one_minus_v, (1, 0), constant_values=1)
    return one_v * c_one

# Log sum exp
def logsumexp(x, axis=0, keepdims=False):
    # return np.log(np.sum(np.exp(x), axis=axis))
    mx = np.max(x, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(x - mx), axis=axis, keepdims=True)) + mx
    if keepdims:
        return out
    else:
        return out.sum(axis=axis)


# In[4]:


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


# In[5]:


# Example:
# rng_key = random.PRNGKey(0)
# mu = dist.Normal(0, 1).sample(rng_key, (3, ))
# sig = dist.Uniform(0, 1).sample(rng_key, (3, ))
# w = np.array([.5, .3, .2])
# x = dist.Normal(0, 1).sample(rng_key, (10, 1))
# lp_x = NormalMixture(mu[None, :], sig[None, :], w[None, :]).log_prob(x[:, None])
# print(lp_x.shape)
# print(lp_x)


# In[6]:


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
    #     Local variables.
    #     label = numpyro.sample('label', dist.Categorical(eta))
    #     numpyro.sample('obs', dist.Normal(mu[label], sigma[label]), obs=y)


# In[7]:


# Read simulated data.
path_to_data = '../../data/sim-data/gmm-data-n200.json'
with open(path_to_data) as f:
  simdata = json.load(f)


# In[8]:


# Convert data to torch.tensor.
y = np.array(simdata['y'])


# In[9]:


get_ipython().run_cell_magic('time', '', '\n# Set random seed for reproducibility.\nrng_key = random.PRNGKey(0)\n\n# Set up NUTS sampler.\nkernel = NUTS(dp_sb_gmm, max_tree_depth=10, target_accept_prob=0.8)\n\n# NOTE: num_leapfrog = trajectory_length / step_size\n# kernel = HMC(dp_sb_gmm, step_size=.01, trajectory_length=1) \n\nmcmc = MCMC(kernel, num_samples=500, num_warmup=500)\nmcmc.run(rng_key, y, 10)')


# In[10]:


# Get posterior samples
posterior_samples = mcmc.get_samples()

# `np.apply_along_axis` not implemented in numpyro?
# TODO: Is there a more efficient way to do this?
posterior_samples['eta'] = np.vstack([stickbreak(v) for v in posterior_samples['v']])


# In[11]:


def plot_param_post(params, param_name, param_full_name, figsize=(12, 4), truth=None):
    plt.figure(figsize=figsize)
    param = params[param_name]

    plt.subplot(1, 2, 1)
    plt.boxplot(param.T, whis=[2.5, 97.5], showmeans=True, showfliers=False)
    plt.xlabel('mixture components')
    plt.ylabel(param_full_name)
    plt.title('95% Credible Intervals for {}'.format(param_full_name))
    if truth is not None:
        for line in truth:
            plt.axhline(line, ls=':')

    plt.subplot(1, 2, 2)
    plt.plot(param);
    plt.xlabel('iterations')
    plt.ylabel(param_full_name)
    plt.title('Trace plot of {}'.format(param_full_name));


# In[12]:


# TODO: How to get log-likelihood?


# In[13]:


plot_param_post(posterior_samples, 'eta', 'mixture weights', truth=simdata['w'])


# In[14]:


plot_param_post(posterior_samples, 'mu', 'mixture means', truth=simdata['mu'])


# In[15]:


plot_param_post(posterior_samples, 'sigma', 'mixture scales', truth=simdata['sig'])


# In[16]:


plt.hist(posterior_samples['alpha'], bins=30, density=True);
plt.xlabel("alpha")
plt.ylabel("density")
plt.title("Posterior distribution of alpha");


# In[ ]:




