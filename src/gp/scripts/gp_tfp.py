#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('echo "Last updated: `date`"')


# Resources:
# - https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GaussianProcessRegressionModel
# - https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GaussianProcess

# In[2]:


# Import libraries.
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

# Alias.
ExponentiatedQuadratic = tfp.math.psd_kernels.ExponentiatedQuadratic

import sys
sys.path.append('../util')
import gp_plot_util

# Default data type for tensorflow tensors.
dtype = np.float64

# Set random seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

# Set number of cores.
# tf.config.threading.set_inter_op_parallelism_threads(4)


# In[3]:


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


# Here we will use the squared exponential covariance function:
# 
# $$
# \alpha^2 \cdot \exp\left\{-\frac{d^2}{2\rho^2}\right\}
# $$
# 
# where $\alpha$ is the amplitude of the covariance, $\rho$ is the length scale which controls how slowly information decays with distance (larger $\rho$ means information about a point can be used for data far away); and $d$ is the distance.

# In[4]:


# Specify GP model
gp_model = tfd.JointDistributionNamed(dict(
    amplitude=tfd.LogNormal(dtype(0), dtype(0.1)),  # amplitude
    length_scale=tfd.LogNormal(dtype(0), dtype(1)),  # length scale
    v=tfd.LogNormal(dtype(0), dtype(1)),  # model sd
    obs=lambda length_scale, amplitude, v: tfd.GaussianProcess(
          kernel=tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude, length_scale),
          index_points=X[..., np.newaxis],
          observation_noise_variance=v)
))

# Run graph to make sure it works.
_ = gp_model.sample()

# Initial values.
initial_state = [
    1e-1 * tf.ones([], dtype=np.float64, name='amplitude'),
    1e-1 * tf.ones([], dtype=np.float64, name='length_scale'),
    1e-1 * tf.ones([], dtype=np.float64, name='v')
]

# Bijectors (from unconstrained to constrained space)
bijectors = [
    tfp.bijectors.Softplus(),  # amplitude
    tfp.bijectors.Softplus(),  # length_scale
    tfp.bijectors.Softplus()  # v
]

# Unnormalized log posterior
def unnormalized_log_posterior(amplitude, length_scale, v):
    return gp_model.log_prob(amplitude=amplitude, length_scale=length_scale, v=v, obs=y)


# In[5]:


# @tf.function(autograph=False, experimental_compile=True)  # Slower?
# @tfp.experimental.nn.util.tfcompile  # no speedup.
@tf.function(autograph=False)
def run_hmc(num_results, num_burnin_steps):
      return tfp.mcmc.sample_chain(
          num_results=num_results,
          num_burnin_steps=num_burnin_steps,
          current_state=initial_state,
          kernel=tfp.mcmc.SimpleStepSizeAdaptation(
              tfp.mcmc.TransformedTransitionKernel(
                  inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                      target_log_prob_fn=unnormalized_log_posterior,
                      step_size=0.01,
                      num_leapfrog_steps=100),
                  bijector=bijectors),
          num_adaptation_steps=num_burnin_steps),
          trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)


# In[6]:


# set random seed
tf.random.set_seed(2)

# Compile
get_ipython().run_line_magic('time', '_ = run_hmc(1, 1)')

# Run
get_ipython().run_line_magic('time', '[amplitudes, length_scales, v], is_accepted = run_hmc(1000, 1000)')

# Print acceptance rate.
print("Acceptance rate: {}".format(np.mean(is_accepted)))

# Collect posterior samples.
hmc_samples = dict(alpha=amplitudes.numpy(), rho=length_scales.numpy(),
                   sigma=np.sqrt(v.numpy()))


# In[7]:


# @tf.function(autograph=False, experimental_compile=True)  # Slower?
# @tfp.experimental.nn.util.tfcompile
@tf.function(autograph=False)
def run_nuts(num_results, num_burnin_steps):
      return tfp.mcmc.sample_chain(
          seed=1,
          num_results=num_results,
          num_burnin_steps=num_burnin_steps,
          current_state=initial_state,
          kernel=tfp.mcmc.SimpleStepSizeAdaptation(
              tfp.mcmc.TransformedTransitionKernel(
                  inner_kernel = tfp.mcmc.NoUTurnSampler(
                      target_log_prob_fn=unnormalized_log_posterior,
                      max_tree_depth=10, step_size=0.1),
                  bijector=bijectors),
          num_adaptation_steps=num_burnin_steps,
          target_accept_prob=0.8),
          trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted)


# In[8]:


# set random seed
tf.random.set_seed(1)

# Compile
get_ipython().run_line_magic('time', '_ = run_nuts(1, 1)')

# Run
get_ipython().run_line_magic('time', '[amplitudes, length_scales, v], is_accepted = run_nuts(1000, 1000)')

# Print acceptance rate.
print("Acceptance rate: {}".format(np.mean(is_accepted)))

# Collect posterior samples.
nuts_samples = dict(alpha=amplitudes.numpy(), rho=length_scales.numpy(),
                    sigma=np.sqrt(v.numpy()))


# In[9]:


# Plot posterior for HMC
gp_plot_util.make_plots(hmc_samples, suffix="HMC",
                        x=X, y=y, x_grid=x_grid, f=f,
                        sigma_true=simdata['sigma'])


# In[10]:


# Plot posterior for NUTS
gp_plot_util.make_plots(nuts_samples, suffix="NUTS",
                        x=X, y=y, x_grid=x_grid, f=f,
                        sigma_true=simdata['sigma'])


# ## ADVI

# In[11]:


# Variational distribution, which approximates the true posterior.
# ADVI is quite sensitive to initial distritbution.
tf.random.set_seed(3)

# Create variational parameters.
qamp_loc = tf.Variable(tf.random.normal([], dtype=dtype) - 1, name='qamp_loc')
qamp_rho = tf.Variable(tf.random.normal([], dtype=dtype) - 1, name='qamp_rho')

qlength_loc = tf.Variable(tf.random.normal([], dtype=dtype), name='qlength_loc')
qlength_rho = tf.Variable(tf.random.normal([], dtype=dtype), name='qlength_rho')

qv_loc = tf.Variable(tf.random.normal([], dtype=dtype), name='qv_loc')
qv_rho = tf.Variable(tf.random.normal([], dtype=dtype), name='qv_rho')

# Create variational distribution.
surrogate_posterior = tfd.JointDistributionNamed(dict(
    amplitude=tfd.LogNormal(qamp_loc, tf.nn.softplus(qamp_rho)),
    length_scale=tfd.LogNormal(qlength_loc, tf.nn.softplus(qlength_rho)),
    v=tfd.LogNormal(qv_loc, tf.nn.softplus(qv_rho))
))

# Function for running ADVI.
def run_advi(sample_size, num_steps):
    return tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=unnormalized_log_posterior,
        surrogate_posterior=surrogate_posterior,
        optimizer=tf.optimizers.Adam(learning_rate=1e-2),
        seed=1,
        sample_size=sample_size,  # ELBO samples.
        num_steps=num_steps)  # Number of iterations to run optimizer. 


# In[12]:


# Fit GP via ADVI.
get_ipython().run_line_magic('time', 'losses = run_advi(sample_size=1, num_steps=2000)')


# In[13]:


# Plot loss.
plt.plot(losses);
plt.ylabel('Loss')
plt.xlabel('Iteration');

# Extract posterior samples from variational distributions.
advi_samples = surrogate_posterior.sample(1000)
advi_samples = dict(alpha=advi_samples['amplitude'].numpy(),
                    rho=advi_samples['length_scale'].numpy(),
                    sigma=np.sqrt(advi_samples['v'].numpy()))


# In[14]:


# Plot posterior for ADVI.
gp_plot_util.make_plots(advi_samples, suffix="ADVI",
                        x=np.array(simdata['x']), y=np.array(simdata['y']),
                        x_grid=simdata['x_grid'], f=simdata['f'],
                        sigma_true=simdata['sigma'])


# In[ ]:




