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
import copy
from scipy.spatial import distance_matrix


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
    real<lower=0> eps;   // amount to add to diagonal of covariance function (for numerical stability)
    
    // hyperparameters for GP covariance function range and scale.
    real m_rho;
    real<lower=0> s_rho;
    real m_alpha;
    real<lower=0> s_alpha;
}

parameters {
    real<lower=0> rho;   // range parameter in GP covariance fn
    real<lower=0> alpha; // covariance scale parameter in GP covariance fn
}

model {
    matrix[N, N] K;   // GP covariance matrix
    matrix[N, N] LK;  // cholesky of GP covariance matrix

    rho ~ lognormal(m_rho, s_rho);  // GP covariance function range parameter
    alpha ~ lognormal(m_alpha, s_alpha);  // GP covariance function scale parameter
   
    // Using exponential quadratic covariance function
    // K(d) = alpha^2 * exp(-d^2 / (2*rho))
    K = cov_exp_quad(to_array_1d(X), alpha, rho); 
    
    // Add small values along diagonal elements for numerical stability.
    for (n in 1:N) {
        K[n, n] = K[n, n] + eps;
    }
        
    LK = cholesky_decompose(K);

    // GP likelihood.
    y ~ multi_normal_cholesky(rep_vector(0.0, N), LK);
}
"""


# In[4]:


get_ipython().run_cell_magic('time', '', '# Compile model. This takes about a minute.\nsm = pystan.StanModel(model_code=gp_model_code)')


# In[100]:


# Generate data.
np.random.seed(1)

# True function.
def f(x):
    return np.sin(3 * x) * np.sin(x) * (-1)**(x > 0)

# Number of observations.
N = 30

# Predictors.
x = np.random.rand(N) * 6 - 3

# Response.
y = f(x)

# Finer grid for plotting true function.
x_ = np.linspace(-3.5, 3.5, 100)
f_ = f(x_)

# Plot data and true function.
plt.scatter(x, y, label='data')
plt.xlabel('x')
plt.ylabel('y = f(x)');
plt.plot(x_, f_, ls=':', c='grey', label='true f(x)')
plt.legend();


# In[78]:


# Data dictionary.
data = dict(y=y, X=x.reshape(N, 1), N=N, D=1, eps=1e-6, m_rho=-2, s_rho=0.1, m_alpha=0, s_alpha=1)


# In[79]:


get_ipython().run_cell_magic('time', '', '# Fit via ADVI.\nvb_fit = sm.vb(data=data, iter=1000, seed=2)\nvb_samples = pystan_vb_extract(vb_fit)')


# In[80]:


# %%time
# # Fit via HMC
# hmc_fit = sm.sampling(data=data, iter=1000, chains=1, warmup=500, thin=1,
#                       seed=1, algorithm='HMC', control=dict(stepsize=0.01, int_time=1))


# In[81]:


# Covariance function (squared exponential)
def cov_fn(d, rho, alpha):
    return alpha ** 2 * np.exp(-0.5 * (d / rho) ** 2)
    
# Function to create gp prediction function.
def gp_predict_maker(y, x, x_new):
    N = x.shape[0]
    N_new = x_new.shape[0]
    M = N + N_new
    xx = np.concatenate((x_new, x)).reshape(M, 1)
    D = distance_matrix(xx, xx)
    
    # Function which takes parameters of covariance function
    # and predicts at new locations.
    def gp_predict(rho, alpha, eps):
        K = cov_fn(D, rho, alpha) + np.eye(M) * eps
        K_new_old = K[:N_new, N_new:]
        K_old_inv = np.linalg.inv(K[N_new:, N_new:])
        C = K_new_old.dot(K_old_inv)
        mu = C.dot(y)
        S = K[:N_new, :N_new] - C.dot(K_new_old.T)
        return np.random.multivariate_normal(mu, S)
    
    return gp_predict


# In[85]:


# Aliais.
# samples = hmc_fit
samples = vb_samples

# Create new locations for prediction.
# But include the data for illustrative purposes.
x_min = np.min(x) - 1
x_max = np.max(x) + 1
x_new = np.linspace(x_min, x_max, 100)
x_new = np.sort(np.concatenate((x_new, x)))

# Create gp predict function.
gp_predict = gp_predict_maker(y, x, x_new)

# Number of posterior samples.
nsamples = len(samples['alpha'])

# Make predictions at new locations.
preds = np.stack([gp_predict(alpha=samples['alpha'][b],
                             rho=samples['rho'][b],
                             eps=data['eps'])
                  for b in range(nsamples)])


# In[99]:


# Function for plotting parameter posterior.
def plot_post(samples, key, bins=None):
    plt.hist(samples[key], density=True, bins=bins)
    plt.xlabel(key)
    plt.ylabel('density')
    plt.title(key);    
    
# Plot parameters posterior.
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_post(samples, 'alpha', bins=30)
plt.subplot(1, 2, 2)
plot_post(samples, 'rho', bins=30)


# In[90]:


# Summarize function posterior.
preds_mean = preds.mean(0)
preds_lower = np.percentile(preds, 2.5, axis=0)
preds_upper = np.percentile(preds, 97.5, axis=0)

# Plot function posterior.
plt.scatter(x, y, c='black', zorder=3, label='data')
plt.fill_between(x_new, preds_upper, preds_lower, alpha=.3, label='95% CI');
plt.plot(x_new, preds.mean(0), lw=2, label="mean fn.")
plt.plot(x_new, f(x_new), label="truth", lw=2, c='red', ls=':')
plt.legend();


# In[ ]:




