#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
print('Last updated: ', datetime.datetime.now(), '(PT)')


# In[2]:


import json
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
# from pyro.optim import Adam
# from pyro.infer import SVI, Trace_ELBO
# from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
# from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS, HMC
import torch
import torch.distributions.constraints as constraints
from torch.nn.functional import pad


# In[3]:


# Stick break function
def stickbreak(v):
    # cumprod_one_minus_v = torch.log1p(-v).cumsum(-1).exp()
    cumprod_one_minus_v = torch.cumprod(1 - v, dim=-1)
    one_v = pad(v, (0, 1), value=1)
    c_one = pad(cumprod_one_minus_v, (1, 0), value=1)
    return one_v * c_one


# In[4]:


# NOTE: This resulted in poor inference.
class GMM(dist.Distribution):
    def __init__(self, mu, sigma, w):
        self.mu = mu
        self.sigma = sigma
        self.w = w
    
    def sample(self, sample_shape=()):
        return torch.zeros((1, ) + sample_shape)
    
    def log_prob(self, y, dim=-1):
        lp = dist.Normal(self.mu, self.sigma).log_prob(y) + torch.log(self.w)
        return lp.logsumexp(dim=dim)

# Example:
# mu = torch.randn(1, 3)
# sigma = torch.rand(1, 3)
# w = stickbreak(torch.rand(1, 2))
# y = torch.randn(10, )
# GMM(mu, sigma, w).log_prob(y[:, None])


# In[5]:


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
        # For marginalized version.
        # pyro.sample('obs', GMM(mu[None, :], sigma[None, :], eta[None, :]), obs=y[:, None])
        # Local variables.
        label = pyro.sample('label', dist.Categorical(eta))
        pyro.sample('obs', dist.Normal(mu[label], sigma[label]), obs=y)


# In[6]:


# Read simulated data.
path_to_data = '../../data/sim-data/gmm-data-n200.json'
with open(path_to_data) as f:
  simdata = json.load(f)


# In[7]:


# Convert data to torch.tensor.
y = torch.tensor(simdata['y'])


# In[8]:


# Set random seed for reproducibility.
pyro.set_rng_seed(2)

# Set up HMC sampler.
kernel = HMC(dp_sb_gmm, step_size=0.01, num_steps=100, target_accept_prob=0.8)
mcmc = MCMC(kernel, num_samples=500, warmup_steps=500)
mcmc.run(y, 10)

# 06:13 for marginalized version.

# For brevity, inference not included in this notebook. 


# In[9]:


# Set random seed for reproducibility.
pyro.set_rng_seed(1)

# Set up NUTS sampler.
kernel = NUTS(dp_sb_gmm, target_accept_prob=0.8)
mcmc = MCMC(kernel, num_samples=500, warmup_steps=500)
mcmc.run(y, 10)

# 44:07 for marginalized version.


# In[10]:


# Get posterior samples
posterior_samples = mcmc.get_samples()
posterior_samples['eta'] = stickbreak(posterior_samples['v'])


# In[11]:


def plot_param_post(params, param_name, param_full_name, figsize=(12, 4), truth=None):
    plt.figure(figsize=figsize)
    param = params[param_name].numpy()

    plt.subplot(1, 2, 1)
    plt.boxplot(param, whis=[2.5, 97.5], showmeans=True, showfliers=False)
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


plot_param_post(posterior_samples, 'eta', 'mixture weights', truth=simdata['w'])


# In[13]:


plot_param_post(posterior_samples, 'mu', 'mixture means', truth=simdata['mu'])


# In[14]:


plot_param_post(posterior_samples, 'sigma', 'mixture scales', truth=simdata['sig'])


# In[15]:


plt.hist(posterior_samples['alpha'], bins=30, density=True);
plt.xlabel("alpha")
plt.ylabel("density")
plt.title("Posterior distribution of alpha");


# In[ ]:




