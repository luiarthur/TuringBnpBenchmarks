#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
print('Last updated:', datetime.datetime.now(), '(PT)')


# In[2]:


import pystan
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
sys.path.append('../util')
from pystan_vb_extract import pystan_vb_extract
import copy


# In[3]:


model = """
data {
  int<lower=0> K;  // Number of cluster
  int<lower=0> N;  // Number of observations
  real y[N];  // observations
  real<lower=0> alpha_shape;
  real<lower=0> alpha_rate;
  real<lower=0> sigma_shape;
  real<lower=0> sigma_rate;
}

parameters {
  real mu[K]; // cluster means
  // real <lower=0,upper=1> v[K - 1];  // stickbreak components
  vector<lower=0,upper=1>[K - 1] v;  // stickbreak components
  real<lower=0> sigma[K];  // error scale
  real<lower=0> alpha;  // hyper prior DP(alpha, base)
}

transformed parameters {
  simplex[K] eta;
  vector<lower=0,upper=1>[K - 1] cumprod_one_minus_v;

  cumprod_one_minus_v = exp(cumulative_sum(log1m(v)));
  eta[1] = v[1];
  eta[2:(K-1)] = v[2:(K-1)] .* cumprod_one_minus_v[1:(K-2)];
  eta[K] = cumprod_one_minus_v[K - 1];
}

model {
  real ps[K];
  // real alpha = 1;
  
  alpha ~ gamma(alpha_shape, alpha_rate);  // mean = a/b = shape/rate 
  sigma ~ gamma(sigma_shape, sigma_rate);
  mu ~ normal(0, 3);
  v ~ beta(1, alpha);

  for(i in 1:N){
    for(k in 1:K){
      ps[k] = log(eta[k]) + normal_lpdf(y[i] | mu[k], sigma[k]);
    }
    target += log_sum_exp(ps);
  }
}

generated quantities {
  real ll;
  real ps_[K];
  
  ll = 0;
  for(i in 1:N){
    for(k in 1:K){
      ps_[k] = log(eta[k]) + normal_lpdf(y[i] | mu[k], sigma[k]);
    }
    ll += log_sum_exp(ps_);
  }  
}
"""


# In[4]:


# Compile the model.
get_ipython().run_line_magic('time', 'sm = pystan.StanModel(model_code=model)')


# In[5]:


# Read simulated data.
path_to_data = '../../data/sim-data/gmm-data-n200.json'
with open(path_to_data) as f:
    simdata = json.load(f)
    
# Create data dictionary.
data = dict(y=simdata['y'], K=10, N=len(simdata['y']),
            # alpha_shape=1, alpha_rate=10, sigma_shape=20, sigma_rate=100)  # this works, but seems coerced
            alpha_shape=1, alpha_rate=10, sigma_shape=1, sigma_rate=10)


# In[6]:


def init_prior(model, data, seed=None, iter=10000, adapt_iter=1000):
    prior_data = copy.deepcopy(data)
    prior_data['y'] = []
    prior_data['N'] = 0
    prior_sample = model.vb(data=prior_data, iter=iter, seed=seed, algorithm='meanfield',
                            adapt_iter=adapt_iter, init=0, output_samples=1)
    prior_init = pystan_vb_extract(prior_sample)
    prior_init = dict([(k, prior_init[k][0]) for k in prior_init])
    return prior_init


# In[7]:


# Approximate posterior via ADVI: 1.2s
# - ADVI is sensitive to starting values. Should run several times and pick run 
#   that has best fit (e.g. highest ELBO / logliklihood).
# - Variational inference works better with more data. Inference is less accurate
#   with small datasets, due to the variational approximation.

seed_ll = dict()
for seed in range(1, 21):
    prior_init = init_prior(sm, data, seed=seed)
    fit = sm.vb(data=data, iter=500, seed=seed, algorithm='meanfield', adapt_iter=100, verbose=False, init=prior_init)
    mean_loglike = pystan_vb_extract(fit)['ll'].mean()
    print("Seed: {} | Mean loglike: {}".format(seed, mean_loglike))
    seed_ll[seed] = mean_loglike

# Print best seed.    
best_seed = sorted(seed_ll.items(), key=lambda k: k[1])[-1][0]
print('Best seed: {}'.format(best_seed))


# In[8]:


prior_init = init_prior(sm, data, seed=best_seed)
fit = sm.vb(data=data, iter=1000, seed=best_seed, algorithm='meanfield', adapt_iter=1000, verbose=False, init=prior_init)


# In[9]:


# Plot mu, eta
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.boxplot(pystan_vb_extract(fit)['mu'], showmeans=True, showfliers=False, whis=[2.5, 97.5]);
plt.ylabel('mu')
plt.xlabel('mixture means')
for line in simdata['mu']:
    plt.axhline(line, ls=':');
    
plt.subplot(2, 2, 2)
plt.boxplot(pystan_vb_extract(fit)['eta'], showmeans=True, showfliers=False, whis=[2.5, 97.5]);
plt.ylabel('eta')
plt.xlabel('mixture weights')
for line in simdata['w']:
    plt.axhline(line, ls=':');
    
plt.subplot(2, 2, 3)
plt.boxplot(pystan_vb_extract(fit)['sigma'], showmeans=True, showfliers=False, whis=[2.5, 97.5]);
plt.ylabel('sigma')
plt.xlabel('mixture scales')
for line in simdata['sig']:
    plt.axhline(line, ls=':');
    
plt.subplot(2, 2, 4)
plt.hist(pystan_vb_extract(fit)['alpha'], bins=30, density=True);
plt.xlabel('alpha')
plt.ylabel('density')

plt.tight_layout();


# In[10]:


# MCMC setup

# Number of burn in iterations
burn = 500

# Number of sampels to keep
nsamples = 500

# Number of MCMC (HMC / NUTS) iterations in total
niters = burn + nsamples


# In[11]:


get_ipython().run_cell_magic('time', '', "\n# Sample from posterior via HMC: 53s\n# NOTE: num_leapfrog = int_time / stepsize.\nfit = sm.sampling(data=data, iter=niters, chains=1, warmup=burn, thin=1, seed=1,\n                  algorithm='HMC', control=dict(stepsize=0.01, int_time=1))")


# In[12]:


get_ipython().run_cell_magic('time', '', '\n# Sample from posterior via NUTS: 1m 57s\nfit = sm.sampling(data=data, iter=niters, chains=1, warmup=burn, thin=1, seed=1)')


# In[13]:


def plot_param_post(samples, param_name, param_full_name, figsize=(12, 4), truth=None):
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.boxplot(samples[param_name], whis=[2.5, 97.5], showmeans=True, showfliers=False)
    plt.xlabel('mixture components')
    plt.ylabel(param_full_name)
    plt.title('95% Credible Intervals for {}'.format(param_full_name))
    
    if truth is not None:
        for line in truth:
            plt.axhline(line, ls=":")

    plt.subplot(1, 2, 2)
    plt.plot(samples[param_name]);
    plt.xlabel('iterations')
    plt.ylabel(param_full_name)
    plt.title('Trace plot of {}'.format(param_full_name));


# In[14]:


plot_param_post(fit, 'eta', 'mixture weights (eta)', truth=simdata['w'])


# In[15]:


plot_param_post(fit, 'mu', 'mixture means (mu)', truth=simdata['mu'])


# In[16]:


plot_param_post(fit, 'sigma', 'mixture scales (sigma)', truth=simdata['sig'])


# In[17]:


# Plot trace of log likelihood (up to proportionality constant)
plt.plot(fit['lp__'])
plt.xlabel("Iterations (post warmup)")
plt.ylabel("Log likelihood (scaled)");
plt.title('Trace plot of log likelihood (scaled)');


# In[18]:


# Plot distribution of alpha
plt.hist(fit['alpha'], bins=30, density=True);
plt.xlabel('alpha')
plt.ylabel('density')
plt.title('Posterior distribution of DP mass parameter (alpha)');


# In[ ]:




