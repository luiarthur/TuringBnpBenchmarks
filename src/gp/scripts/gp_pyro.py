#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('echo "Last updated: `date`"')


# In[2]:


# Libraries
import json
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS, HMC
import torch
import torch.distributions.constraints as constraints
from torch.nn.functional import pad
from tqdm import trange

# For ADVI
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

# For GP
import pyro.contrib.gp as gp

import sys
sys.path.append('../util')
import gp_plot_util

# See:
# http://docs.pyro.ai/en/stable/contrib.gp.html
# https://pyro.ai/examples/gp.html


# In[3]:


# Read data.
path_to_data = '../data/gp-data-N30.json'
simdata = json.load(open(path_to_data))

# Store data as torch.tensors.
X = torch.tensor(simdata['x']).reshape(-1, 1)
y = torch.tensor(simdata['y'])
x_grid = torch.tensor(simdata['x_grid'])
f = torch.tensor(simdata['f'])

# Plot data and true function.
plt.scatter(X, y, label='data')
plt.plot(x_grid, f, ls=':', c='grey', label='true f(x)')
plt.xlabel('x')
plt.ylabel('y = f(x)')
plt.legend();


# In[4]:


def make_gp_model(X, y,
                  length_prior=dist.LogNormal(0.0, 1.0),
                  variance_prior=dist.LogNormal(0.0, 0.1),
                  noise_prior=dist.LogNormal(0.0, 1.0)):
    
    # Define squared exponential covariance function.
    cov_fn = gp.kernels.RBF(input_dim=1)

    # Define GP regression model.
    gpr = gp.models.GPRegression(X, y, cov_fn)

    # Place priors on GP covariance function parameters.
    gpr.kernel.lengthscale = pyro.nn.PyroSample(length_prior)
    gpr.kernel.variance = pyro.nn.PyroSample(variance_prior)
    gpr.noise = pyro.nn.PyroSample(noise_prior)
    
    return gpr


# In[5]:


get_ipython().run_cell_magic('time', '', "\n### HMC ###\npyro.clear_param_store()\n\n# Set random seed for reproducibility.\npyro.set_rng_seed(1)\n\n# Make GP model for HMC\nhmc_gpr = make_gp_model(X, y)\n\n# Set up HMC sampler.\n# kernel = HMC(hmc_gpr.model, step_size=0.01, trajectory_length=1,\n#              adapt_step_size=True, adapt_mass_matrix=True)  # 14s\nkernel = HMC(hmc_gpr.model, step_size=0.01, trajectory_length=1,\n             adapt_step_size=False, adapt_mass_matrix=False, jit_compile=True)\nhmc = MCMC(kernel, num_samples=1000, warmup_steps=1000)\nhmc.run()\n\n# Get posterior samples\nhmc_posterior_samples = hmc.get_samples()\nhmc_posterior_samples = dict(rho=hmc_posterior_samples['kernel.lengthscale'].numpy(),\n                             alpha=hmc_posterior_samples['kernel.variance'].sqrt().numpy(),\n                             sigma=hmc_posterior_samples['noise'].sqrt().numpy())")


# In[6]:


get_ipython().run_cell_magic('time', '', "\n### NUTS ###\npyro.clear_param_store()\n\n# Set random seed for reproducibility.\npyro.set_rng_seed(1)\n\n# Make GP model for NUTS\nnuts_gpr = make_gp_model(X, y)\n\n# Set up NUTS sampler.\nkernel = NUTS(nuts_gpr.model, target_accept_prob=0.8, jit_compile=True)\nnuts = MCMC(kernel, num_samples=1000, warmup_steps=1000)\n%time nuts.run()\n\n# Get posterior samples\nnuts_posterior_samples = nuts.get_samples()\nnuts_posterior_samples = dict(rho=nuts_posterior_samples['kernel.lengthscale'].numpy(),\n                              alpha=nuts_posterior_samples['kernel.variance'].sqrt().numpy(),\n                              sigma=nuts_posterior_samples['noise'].sqrt().numpy())")


# In[7]:


# Plot posterior for HMC
gp_plot_util.make_plots(hmc_posterior_samples, suffix="HMC",
                        x=X.flatten(), y=y, x_grid=x_grid, f=f, sigma_true=simdata['sigma'])


# In[8]:


# Plot posterior for NUTS
gp_plot_util.make_plots(nuts_posterior_samples, suffix="NUTS",
                        x=X.flatten(), y=y, x_grid=x_grid, f=f, sigma_true=simdata['sigma'])


# In[9]:


get_ipython().run_cell_magic('time', '', '\n### MAP ###\n# Can\'t get posterior samples for ADVI?\n\n# Clear parameter cache.\npyro.clear_param_store()\n\n# Set random seed for reproducibility.\npyro.set_rng_seed(3)\n\n# Make GP model for MAP\nmap_gpr = make_gp_model(X, y)\n\n# Find MAP estimator\noptimizer = torch.optim.Adam(map_gpr.parameters(), lr=5e-2)\nloss_fn = pyro.infer.Trace_ELBO().differentiable_loss\nlosses = []\nnum_steps = 500\nfor i in trange(num_steps):\n    optimizer.zero_grad()\n    loss = loss_fn(map_gpr.model, map_gpr.guide)\n    loss.backward()\n    optimizer.step()\n    losses.append(loss.item())\n    \n# Plot loss\nplt.plot(losses)\nplt.xlabel("Iteration")\nplt.ylabel("loss");\n\n# This \nmap_gpr.set_mode(\'guide\')\nmap_samples = dict(alpha=[map_gpr.kernel.variance.sqrt().item() for _ in range(10)],\n                   rho=[map_gpr.kernel.lengthscale.item() for _ in range(10)],\n                   sigma=[map_gpr.noise.sqrt().item() for _ in range(10)])')


# In[10]:


# Plot posterior for MAP
gp_plot_util.make_plots(map_samples, suffix="MAP",
                        x=X.flatten(), y=y, x_grid=x_grid, f=f, sigma_true=simdata['sigma'])


# In[ ]:




