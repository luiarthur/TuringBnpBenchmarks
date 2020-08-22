#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('echo "Last updated: `date`"')


# In[73]:


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
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, JitTrace_ELBO
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

# For GP
import pyro.contrib.gp as gp

import sys
sys.path.append('../util')
import gp_plot_util

# Default to double precision for torch objects.
torch.set_default_dtype(torch.float64)

# See also:
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


# In[70]:


# Model definition.
def sq_exp_kernel(d, alpha, rho):
    return alpha * alpha * torch.exp(-0.5 * torch.pow(d / rho, 2))

def sq_exp_kernel_matrix(X, alpha, rho):
    D = torch.cdist(X, X)
    return sq_exp_kernel(D, alpha, rho)

def gp_model(X, y):
    N = X.shape[0]
    
    # Priors for kernel parameters.
    alpha = pyro.sample("alpha", dist.LogNormal(0., 0.1))
    rho = pyro.sample("rho", dist.LogNormal(0., 1.))
    sigma = pyro.sample("sigma", dist.LogNormal(0., 1.))
    
    # Covariance matrix. 
    K = sq_exp_kernel_matrix(X, alpha, rho) + torch.eye(N) * sigma * sigma
    L = torch.cholesky(K)
    
    # Marginal likelihood.
    pyro.sample('obs',
                dist.MultivariateNormal(torch.zeros(N), scale_tril=L),
                obs=y)


# In[64]:


get_ipython().run_cell_magic('time', '', '\n### HMC ###\npyro.clear_param_store()\n\n# Set random seed for reproducibility.\npyro.set_rng_seed(1)\n\n# Set up HMC sampler.\nhmc = MCMC(HMC(gp_model, step_size=0.01, trajectory_length=1,\n               adapt_step_size=False, adapt_mass_matrix=False,\n               jit_compile=True),\n           num_samples=1000, warmup_steps=1000)\nhmc.run(X, y)\n\n# Get posterior samples\nhmc_posterior_samples = hmc.get_samples()\nhmc_posterior_samples = {k: hmc_posterior_samples[k].numpy()\n                         for k in hmc_posterior_samples}')


# In[65]:


get_ipython().run_cell_magic('time', '', '\n### NUTS ###\npyro.clear_param_store()\n\n# Set random seed for reproducibility.\npyro.set_rng_seed(1)\n\n# Set up NUTS sampler.\nnuts = MCMC(NUTS(gp_model, target_accept_prob=0.8, jit_compile=True),\n            num_samples=1000, warmup_steps=1000)\n%time nuts.run(X, y)\n\n# Get posterior samples\nnuts_posterior_samples = nuts.get_samples()\nnuts_posterior_samples = {k: nuts_posterior_samples[k].numpy()\n                          for k in nuts_posterior_samples}')


# In[66]:


# Plot posterior for HMC
gp_plot_util.make_plots(hmc_posterior_samples, suffix="HMC",
                        x=X.flatten(), y=y, x_grid=x_grid, f=f, sigma_true=simdata['sigma'])


# In[67]:


# Plot posterior for NUTS
gp_plot_util.make_plots(nuts_posterior_samples, suffix="NUTS",
                        x=X.flatten(), y=y, x_grid=x_grid, f=f, sigma_true=simdata['sigma'])


# In[91]:


get_ipython().run_cell_magic('time', '', "# ADVI\npyro.clear_param_store()  # clear global parameter cache\npyro.set_rng_seed(1)  # set random seed\n\n# Automatically define variational distribution (a mean field guide).\nguide = AutoDiagonalNormal(gp_model)\n\n# Create SVI object for optimization.\nsvi = SVI(gp_model, guide, Adam({'lr': 1e-2}), JitTrace_ELBO())\n\n# Do 1000 gradient steps.\nadvi_loss = []\nfor step in trange(1000):\n    advi_loss.append(svi.step(X, y.double()))\n    \n# Bijector for advi samples.\ndef biject(samples):\n    return dict(alpha=samples[:, 0].exp().numpy(),\n                rho=samples[:, 1].exp().numpy(),\n                sigma=samples[:, 2].exp().numpy())\n\n# Get ADVI samples in constrained space.\nadvi_posterior_samples = biject(guide.get_posterior().sample((1000, )))   ")


# In[79]:


# Plot posterior for ADVI
gp_plot_util.make_plots(advi_posterior_samples, suffix="ADVI",
                        x=X.flatten(), y=y, x_grid=x_grid, f=f, sigma_true=simdata['sigma'])


# In[ ]:




