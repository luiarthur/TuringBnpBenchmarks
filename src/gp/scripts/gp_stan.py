#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('echo "Last updated:" `date`')


# # Fitting regular GP in STAN
# 
# This notebook demonstrates how a GP is specified and sampled from in STAN.

# In[2]:


import pystan
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
sys.path.append('../util')
from pystan_vb_extract import pystan_vb_extract
import gp_plot_util


# In[3]:


# Define GP model.

# See for an explaination of the parameters (rho, alpha) and the convariance function used.
# https://mc-stan.org/docs/2_19/stan-users-guide/gaussian-process-regression.html

gp_model_code = """
data {
    int D;               // number of features (dimensions of X)
    int N;               // number of observations
    vector[N] y;         // response
    matrix[N, D] X;      // predictors
    
    // hyperparameters for GP covariance function range and scale.
    real m_rho;
    real<lower=0> s_rho;
    real m_alpha;
    real<lower=0> s_alpha;
    real m_sigma;
    real<lower=0> s_sigma;
}

transformed data {
    // GP mean function.
    vector[N] mu = rep_vector(0, N);
}

parameters {
    real<lower=0> rho;   // range parameter in GP covariance fn
    real<lower=0> alpha; // covariance scale parameter in GP covariance fn
    real<lower=0> sigma;   // model sd
}

model {
    matrix[N, N] K;   // GP covariance matrix
    matrix[N, N] LK;  // cholesky of GP covariance matrix

    rho ~ lognormal(m_rho, s_rho);  // GP covariance function range parameter
    alpha ~ lognormal(m_alpha, s_alpha);  // GP covariance function scale parameter
    sigma ~ lognormal(m_sigma, s_sigma);  // model sd.
   
    // Using exponential quadratic covariance function
    // K(d) = alpha^2 * exp(-0.5 * (d/rho)^2)
    K = cov_exp_quad(to_array_1d(X), alpha, rho); 
    
    // Add small values along diagonal elements for numerical stability.
    for (n in 1:N) {
        K[n, n] = K[n, n] + sigma^2;
    }
        
    // Cholesky of K (lower triangle).
    LK = cholesky_decompose(K);

    // GP likelihood.
    y ~ multi_normal_cholesky(mu, LK);
}
"""


# In[4]:


get_ipython().run_cell_magic('time', '', '# Compile model. This takes about a minute.\nsm = pystan.StanModel(model_code=gp_model_code)')


# In[5]:


# Read data.
path_to_data = '../data/gp-data-N30.json'
simdata = json.load(open(path_to_data))

# Plot data and true function.
plt.scatter(simdata['x'], simdata['y'], label='data')
plt.plot(simdata['x_grid'], simdata['f'], ls=':', c='grey', label='true f(x)')
plt.xlabel('x')
plt.ylabel('y = f(x)')
plt.legend();


# In[6]:


# Data dictionary.
N = len(simdata['x'])
data = dict(y=simdata['y'], X=np.reshape(simdata['x'], (N, 1)), N=N, D=1,
            m_rho=0, s_rho=1.0, m_alpha=0, s_alpha=0.1, m_sigma=0, s_sigma=1)


# In[7]:


get_ipython().run_cell_magic('time', '', '# Fit via ADVI.\nvb_fit = sm.vb(data=data, iter=2000, seed=2, grad_samples=1, elbo_samples=1)\nvb_samples = pystan_vb_extract(vb_fit)')


# In[8]:


get_ipython().run_cell_magic('time', '', "# Fit via HMC\nhmc_fit = sm.sampling(data=data, iter=2000, chains=1, warmup=1000, thin=1,\n                      seed=1, algorithm='HMC', control=dict(stepsize=0.01, int_time=1))")


# In[9]:


get_ipython().run_cell_magic('time', '', '# Fit via NUTS\nnuts_fit = sm.sampling(data=data, iter=2000, chains=1, warmup=1000, thin=1, seed=1)')


# In[10]:


gp_plot_util.make_plots(vb_samples, suffix="ADVI",
                        x=np.array(simdata['x']), y=np.array(simdata['y']),
                        x_grid=simdata['x_grid'], f=simdata['f'], sigma_true=simdata['sigma'])


# In[11]:


gp_plot_util.make_plots(hmc_fit, suffix="HMC",
                        x=np.array(simdata['x']), y=np.array(simdata['y']),
                        x_grid=simdata['x_grid'], f=simdata['f'], sigma_true=simdata['sigma'])


# In[12]:


gp_plot_util.make_plots(nuts_fit, suffix="NUTS",
                        x=np.array(simdata['x']), y=np.array(simdata['y']),
                        x_grid=simdata['x_grid'], f=simdata['f'], sigma_true=simdata['sigma'])


# In[ ]:




