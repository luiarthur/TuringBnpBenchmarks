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


# In[50]:


# Read data.
path_to_data = '../data/gp-data-N30.json'
simdata = json.load(open(path_to_data))

# Plot data and true function.
plt.scatter(simdata['x'], simdata['f'], label='data')
plt.plot(simdata['x_true'], simdata['f_true'], ls=':', c='grey', label='true f(x)')
plt.xlabel('x')
plt.ylabel('y = f(x)')
plt.legend();

# Store data as torch.tensors.
X = torch.tensor(simdata['x']).reshape(len(simdata['x']), 1)
y = torch.tensor(simdata['f'])


# In[63]:


def make_gp_model(X, y, noise=torch.tensor(1e-3).sqrt(),
                  length_prior=dist.LogNormal(-2, 0.1),
                  variance_prior=dist.LogNormal(0.0, 1.0)):
    
    # Define squared exponential covariance function.
    cov_fn = gp.kernels.RBF(input_dim=1)

    # Define GP regression model.
    gpr = gp.models.GPRegression(X, y, cov_fn, noise=noise)

    # Place priors on GP covariance function parameters.
    gpr.kernel.lengthscale = pyro.nn.PyroSample(length_prior)
    gpr.kernel.variance = pyro.nn.PyroSample(variance_prior)
    
    return gpr


# In[53]:


get_ipython().run_cell_magic('time', '', "\n### HMC ###\npyro.clear_param_store()\n\n# Set random seed for reproducibility.\npyro.set_rng_seed(1)\n\n# Make GP model for HMC\nhmc_gpr = make_gp_model(X, y)\n\n# Set up HMC sampler.\nkernel = HMC(hmc_gpr.model, step_size=0.01, trajectory_length=1, target_accept_prob=0.8,\n             adapt_step_size=False, adapt_mass_matrix=False)\nhmc = MCMC(kernel, num_samples=1000, warmup_steps=1000)\nhmc.run()\n\n# Get posterior samples\nhmc_posterior_samples = hmc.get_samples()\nhmc_posterior_samples = dict(rho=hmc_posterior_samples['kernel.lengthscale'].numpy(),\n                             alpha=hmc_posterior_samples['kernel.variance'].sqrt().numpy())")


# In[70]:


get_ipython().run_cell_magic('time', '', "\n### NUTS ###\npyro.clear_param_store()\n\n# Set random seed for reproducibility.\npyro.set_rng_seed(1)\n\n# Make GP model for NUTS\nnuts_gpr = make_gp_model(X, y)\n\n# Set up NUTS sampler.\nkernel = NUTS(nuts_gpr.model, target_accept_prob=0.8)\nnuts = MCMC(kernel, num_samples=1000, warmup_steps=1000)\n%time nuts.run()\n\n# Get posterior samples\nnuts_posterior_samples = nuts.get_samples()\nnuts_posterior_samples = dict(rho=nuts_posterior_samples['kernel.lengthscale'].numpy(),\n                              alpha=nuts_posterior_samples['kernel.variance'].sqrt().numpy())")


# In[71]:


# Plot posterior for NUTS
gp_plot_util.make_plots(nuts_posterior_samples, suffix="NUTS",
                        x=np.array(simdata['x']), y=np.array(simdata['f']),
                        x_true=simdata['x_true'], f_true=simdata['f_true'])


# In[75]:


# Plot posterior for HMC
gp_plot_util.make_plots(hmc_posterior_samples, suffix="HMC",
                        x=np.array(simdata['x']), y=np.array(simdata['f']),
                        x_true=simdata['x_true'], f_true=simdata['f_true'])


# In[208]:


get_ipython().run_cell_magic('time', '', '\n### MAP ###\n# Can\'t get posterior samples for ADVI?\n\n# Clear parameter cache.\npyro.clear_param_store()\n\n# Set random seed for reproducibility.\npyro.set_rng_seed(2)\n\n# Make GP model for MAP\nmap_gpr = make_gp_model(X, y)\n\n# Find MAP estimator\noptimizer = torch.optim.Adam(map_gpr.parameters(), lr=5e-2)\nloss_fn = pyro.infer.Trace_ELBO().differentiable_loss\nlosses = []\nnum_steps = 2000\nfor i in trange(num_steps):\n    optimizer.zero_grad()\n    loss = loss_fn(map_gpr.model, map_gpr.guide)\n    loss.backward()\n    optimizer.step()\n    losses.append(loss.item())\n    \n# Plot loss\nplt.plot(losses)\nplt.xlabel("Iteration")\nplt.ylabel("loss");\n\n# This \nmap_gpr.set_mode(\'guide\')\nmap_samples = dict(alpha=[map_gpr.kernel.variance.sqrt().item() for _ in range(10)],\n                   rho=[map_gpr.kernel.lengthscale.item() for _ in range(10)])')


# In[209]:


# Plot posterior for MAP
gp_plot_util.make_plots(map_samples, suffix="MAP",
                        x=np.array(simdata['x']), y=np.array(simdata['f']),
                        x_true=simdata['x_true'], f_true=simdata['f_true'])


# In[ ]:




