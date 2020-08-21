# Import libraries.
import os
import sys
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
from sklearn.datasets import make_moons

# For ADVI
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

# For GP
import pyro.contrib.gp as gp

# Default to double precision for torch objects.
torch.set_default_dtype(torch.float64)


# Make data.
X, y = make_moons(n_samples=50, shuffle=True, noise=0.1, random_state=1)
X = torch.from_numpy(X)
y = torch.from_numpy(y)


# Model definition.
def sq_exp_kernel(d, alpha, rho):
    return alpha * alpha * torch.exp(-0.5 * torch.pow(d / rho, 2))

def compute_f(alpha, rho, beta, eta, X):
    N = X.shape[0]
    D = torch.cdist(X, X)
    K = sq_exp_kernel(D, alpha, rho) + torch.eye(N) * 1e-6
    L = K.cholesky()
    return L.matmul(eta) + beta

# GP Binary Classifier.
def gpc(X, y):
    N = y.shape[0]
    
    # Priors.
    alpha = pyro.sample('alpha', dist.LogNormal(0, 1))
    rho = pyro.sample('rho', dist.LogNormal(0, 1))
    beta = pyro.sample('beta', dist.Normal(0, 1))

    with pyro.plate('latent_response', N):
        eta = pyro.sample('eta', dist.Normal(0, 1))

    # Latent function.
    f = compute_f(alpha, rho, beta, eta, X)
   
    with pyro.plate('response', N):
        pyro.sample('obs', dist.Bernoulli(logits=f), obs=y)


# HMC
pyro.clear_param_store()  # clear global parameter cache.
pyro.set_rng_seed(2) # set random number generator seed.
hmc = MCMC(HMC(gpc, step_size=0.05, trajectory_length=1,
               adapt_step_size=False, adapt_mass_matrix=False,
               jit_compile=True),
           num_samples=500, warmup_steps=500)  # sampler setup.
hmc.run(X, y.double())  # run mcmc
hmc_posterior_samples = hmc.get_samples()  # get posterior samples.


# NUTS
pyro.clear_param_store() 
pyro.set_rng_seed(2)
nuts = MCMC(NUTS(gpc, target_accept_prob=0.8, max_tree_depth=10,
                 jit_compile=True),
            num_samples=500, warmup_steps=500)
nuts.run(X, y.double())
nuts_posterior_samples = nuts.get_samples()



# ADVI
pyro.clear_param_store()  # clear global parameter cache
pyro.set_rng_seed(1)  # set random seed

# Automatically define variational distribution (a mean field guide).
guide = AutoDiagonalNormal(gpc)

# Create SVI object for optimization.
svi = SVI(gpc, guide, Adam({'lr': 1e-2}), TraceEnum_ELBO())

# Do 1000 gradient steps.
advi_loss = []
for step in trange(1000):
    advi_loss.append(svi.step(X, y.double()))

# NOTE: See notebook to see full example.
