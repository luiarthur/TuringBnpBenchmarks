#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('echo "Last updated: `date`"')


# In[4]:


import json
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC
import numpy as onp

import sys
sys.path.append('../util')
import gp_plot_util


# In[11]:


# One-dimensional squared exponential kernel with diagonal noise term.
def squared_exp_cov_1D(X, variance, lengthscale):
    deltaXsq = np.power((X[:, None] - X) / lengthscale, 2.0)
    K = variance * np.exp(-0.5 * deltaXsq)
    return K

# GP model.
def GP(X, y):
    # Set informative log-normal priors on kernel hyperparameters.
    variance = numpyro.sample("kernel_var", dist.LogNormal(0.0, 0.1))
    lengthscale = numpyro.sample("kernel_length", dist.LogNormal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 1.0))

    # Compute kernel
    K = squared_exp_cov_1D(X, variance, lengthscale)
    K += np.eye(X.shape[0]) * np.power(sigma, 2)

    # Sample y according to the standard gaussian process formula
    numpyro.sample("y", dist.MultivariateNormal(loc=np.zeros(X.shape[0]),
                                                covariance_matrix=K), obs=y)


# In[12]:


# Read data.
path_to_data = '../data/gp-data-N30.json'
simdata = json.load(open(path_to_data))

# Store data as torch.tensors.
X = np.array(simdata['x'])
y = np.array(simdata['y'])
x_grid = np.array(simdata['x_grid'])
f = np.array(simdata['f'])

# Plot data and true function.
plt.scatter(X, y, label='data')
plt.plot(x_grid, f, ls=':', c='grey', label='true f(x)')
plt.xlabel('x')
plt.ylabel('y = f(x)')
plt.legend();


# In[13]:


get_ipython().run_cell_magic('time', '', "\n# Set random seed for reproducibility.\nrng_key = random.PRNGKey(0)\n\n# NOTE: num_leapfrog = trajectory_length / step_size\nkernel = HMC(GP, step_size=.01, trajectory_length=1) \n\nhmc = MCMC(kernel, num_samples=1000, num_warmup=1000)\nhmc.run(rng_key, X, y)\n\nhmc_samples = hmc.get_samples()\nhmc_samples = dict(alpha=np.sqrt(hmc_samples['kernel_var']), rho=hmc_samples['kernel_length'],\n                   sigma=hmc_samples['sigma'])")


# In[14]:


get_ipython().run_cell_magic('time', '', "\n# Set random seed for reproducibility.\nrng_key = random.PRNGKey(0)\n\n# Set up NUTS sampler.\nkernel = NUTS(GP, max_tree_depth=10, target_accept_prob=0.8)\n\nnuts = MCMC(kernel, num_samples=1000, num_warmup=1000)\nnuts.run(rng_key, X, y)\n\nnuts_samples = hmc.get_samples()\nnuts_samples = dict(alpha=np.sqrt(nuts_samples['kernel_var']), rho=nuts_samples['kernel_length'],\n                    sigma=nuts_samples['sigma'])")


# In[15]:


# Plot posterior for HMC
gp_plot_util.make_plots(hmc_samples, suffix="HMC",
                        x=X, y=y, x_grid=x_grid, f=f, sigma_true=simdata['sigma'])


# In[17]:


# Plot posterior for NUTS
gp_plot_util.make_plots(hmc_samples, suffix="NUTS",
                        x=X, y=y, x_grid=x_grid, f=f, sigma_true=simdata['sigma'])


# In[ ]:




