#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
print('Last updated: ', datetime.datetime.now(), '(PT)')


# In[63]:


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
    cumprod_one_minus_v = torch.log1p(-v).cumsum(-1).exp()
    one_v = pad(v, (0, 1), value=1)
    c_one = pad(cumprod_one_minus_v, (1, 0), value=1)
    return one_v * c_one


# In[60]:


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
    
    # v = pyro.sample('v', dist.Beta(torch.ones(1, K - 1)), alpha)
    # eta = stickbreak(v)
    
    with pyro.plate('components', K):
        mu = pyro.sample('mu', dist.Normal(0., 3.))
        sigma = pyro.sample('sigma', dist.Gamma(1, 10))

    with pyro.plate('data', N):
        # Local variables.
        label = pyro.sample('label', dist.Categorical(eta))
        pyro.sample('obs', dist.Normal(mu[label], sigma[label]), obs=y)


# In[5]:


# Read simulated data.
path_to_data = '../../data/sim-data/gmm-data-n200.json'
with open(path_to_data) as f:
  simdata = json.load(f)


# In[6]:


# Convert data to torch.tensor.
y = torch.tensor(simdata['y'])


# In[65]:


# # Set random seed for reproducibility.
# pyro.set_rng_seed(1)
# 
# # Set up NUTS sampler.
# kernel = HMC(dp_sb_gmm, step_size=0.01, num_steps=100)
# mcmc = MCMC(kernel, num_samples=100, warmup_steps=100)
# mcmc.run(y, 10)


# In[43]:


# Set random seed for reproducibility.
pyro.set_rng_seed(1)

# Set up NUTS sampler.
kernel = NUTS(dp_sb_gmm)
mcmc = MCMC(kernel, num_samples=500, warmup_steps=500)
mcmc.run(y, 10)


# In[44]:


# Get posterior samples
posterior_samples = mcmc.get_samples()
posterior_samples['eta'] = stickbreak(posterior_samples['v'])


# In[45]:


def plot_param_post(params, param_name, param_full_name, figsize=(12, 4)):
    plt.figure(figsize=figsize)
    param = params[param_name].numpy()

    plt.subplot(1, 2, 1)
    plt.boxplot(param, whis=[2.5, 97.5], showmeans=True, showfliers=False)
    plt.xlabel('mixture components')
    plt.ylabel(param_full_name)
    plt.title('95% Credible Intervals for {}'.format(param_full_name))
    # for wk in  simdata['w']:
    #     plt.axhline(wk)

    plt.subplot(1, 2, 2)
    plt.plot(param);
    plt.xlabel('iterations')
    plt.ylabel(param_full_name)
    plt.title('Trace plot of {}'.format(param_full_name));


# In[46]:


plot_param_post(posterior_samples, 'eta', 'mixture weights')


# In[47]:


plot_param_post(posterior_samples, 'mu', 'mixture means')


# In[48]:


plot_param_post(posterior_samples, 'sigma', 'mixture scales')


# In[59]:


plt.hist(posterior_samples['alpha'], bins=30, density=True);
plt.xlabel("alpha")
plt.ylabel("density")
plt.title("Posterior distribution of alpha");


# In[ ]:




