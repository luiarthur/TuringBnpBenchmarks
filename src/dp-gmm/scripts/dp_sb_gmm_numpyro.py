#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
print('Last updated: ', datetime.datetime.now(), '(PT)')


# In[17]:


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


# In[4]:


def logsumexp(x, axis=0):
    return np.log(np.sum(np.exp(x), axis=axis))

class NormalMixture(dist.Distribution):
    support = constraints.real_vector

    def __init__(self, mu, sigma, w):
        super(NormalMixture, self).__init__(event_shape=(w.shape[-1], ))
        self.mu = mu
        self.sigma = sigma
        self.w = w

    def sample(self, key, sample_shape=()):
        # it is enough to return an arbitrary sample with correct shape
        return np.zeros(sample_shape + self.event_shape)

    def log_prob(self, y):
        x = dist.Normal(self.mu, self.sigma).log_prob(y) + np.log(self.w)
        return logsumexp(x, axis=-1)


# In[5]:


# NormalMixture(np.ones((1, 3)), np.ones((1, 3)), np.array([.5, .3, .2])[None, :]).log_prob(np.zeros((10, 3)))


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
        # Local variables.
        # label = numpyro.sample('label', dist.Categorical(eta))
        # numpyro.sample('obs', dist.Normal(mu[label], sigma[label]), obs=y)
        numpyro.sample('obs', NormalMixture(mu[None, :] , sigma[None, :], eta[None, :]), obs=y[:, None])


# In[7]:


# Read simulated data.
path_to_data = '../../data/sim-data/gmm-data-n200.json'
with open(path_to_data) as f:
  simdata = json.load(f)


# In[8]:


# Convert data to torch.tensor.
y = np.array(simdata['y'])


# In[31]:


get_ipython().run_cell_magic('time', '', '\n# Set random seed for reproducibility.\nrng_key = random.PRNGKey(0)\n\n# Set up NUTS sampler.\nkernel = NUTS(dp_sb_gmm)\n\n# NOTE: num_leapfrog = trajectory_length / step_size\n# kernel = HMC(dp_sb_gmm, step_size=.01, trajectory_length=1) \n\nmcmc = MCMC(kernel, num_samples=500, num_warmup=500)\nmcmc.run(rng_key, y, 10)')


# In[24]:


# Get posterior samples
posterior_samples = mcmc.get_samples()
posterior_samples['eta'] = stickbreak(posterior_samples['v'])


# In[25]:


def plot_param_post(params, param_name, param_full_name, figsize=(12, 4)):
    plt.figure(figsize=figsize)
    param = params[param_name]

    plt.subplot(1, 2, 1)
    plt.boxplot(param.T, whis=[2.5, 97.5], showmeans=True, showfliers=False)
    plt.xlabel('mixture components')
    plt.ylabel(param_full_name)
    plt.title('95% Credible Intervals for {}'.format(param_full_name))

    plt.subplot(1, 2, 2)
    plt.plot(param);
    plt.xlabel('iterations')
    plt.ylabel(param_full_name)
    plt.title('Trace plot of {}'.format(param_full_name));


# In[26]:


# TODO: How to get log-likelihood?


# In[27]:


plot_param_post(posterior_samples, 'eta', 'mixture weights')


# In[28]:


plot_param_post(posterior_samples, 'mu', 'mixture means')


# In[29]:


plot_param_post(posterior_samples, 'sigma', 'mixture scales')


# In[30]:


plt.hist(posterior_samples['alpha'], bins=30, density=True);
plt.xlabel("alpha")
plt.ylabel("density")
plt.title("Posterior distribution of alpha");


# In[ ]:




