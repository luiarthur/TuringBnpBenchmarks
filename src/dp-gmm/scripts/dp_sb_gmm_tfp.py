#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('echo "Late updated:" `date`')


# Resources for learning TFP
# - https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/NoUTurnSampler
# - https://www.tensorflow.org/probability/overview
# - https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc
# - https://www.tensorflow.org/probability/examples/Modeling_with_JointDistribution
# - https://www.tensorflow.org/probability/examples/Bayesian_Gaussian_Mixture_Model
# 
# To better understand event_shape, batch_shape, sample_shape:
# - https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution
# - https://www.youtube.com/watch?v=zWXTpZX4PPo

# In[2]:


import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

# Default data type for tensorflow tensors.
dtype = tf.float64


# In[3]:


# Set random seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)


# In[4]:


# Thanks to Dave Moore for extending this to work with batch dimensions!
# This turns out to be necessary for ADVI to work properly.
def stickbreak(v):
    batch_ndims = len(v.shape) - 1
    cumprod_one_minus_v = tf.math.cumprod(1 - v, axis=-1)
    one_v = tf.pad(v, [[0, 0]] * batch_ndims + [[0, 1]], "CONSTANT", constant_values=1)
    c_one = tf.pad(cumprod_one_minus_v, [[0, 0]] * batch_ndims + [[1, 0]], "CONSTANT", constant_values=1)
    return one_v * c_one

# Example:
# stickbreak(np.random.rand(2, 3))  # Last dimension is the number of sticks breaks.
# Returns a tensor of shape(2, 4).
#
# stickbreak(np.random.rand(2, 3, 4))  # Last dimension is the number of sticks breaks.
# Returns a tensor of shape(2, 3, 5).
#
# The last dimensions sum to 1.
# stickbreak(np.random.randn(2,3,4)).numpy().sum(-1)


# In[5]:


# See: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MixtureSameFamily
# See: https://www.tensorflow.org/probability/examples/Bayesian_Gaussian_Mixture_Model
def create_dp_sb_gmm(nobs, K, dtype=np.float64):
    return tfd.JointDistributionNamed(dict(
        # Mixture means
        mu = tfd.Independent(
            tfd.Normal(np.zeros(K, dtype), 3),
            reinterpreted_batch_ndims=1
        ),
        # Mixture scales
        sigma = tfd.Independent(
            tfd.LogNormal(loc=np.full(K, - 2, dtype), scale=0.5),
            reinterpreted_batch_ndims=1
        ),
        # Mixture weights (stick-breaking construction)
        alpha = tfd.Gamma(concentration=np.float64(1.0), rate=10.0),
        v = lambda alpha: tfd.Independent(
            # NOTE: Dave Moore suggests doing this instead, to ensure 
            # that a batch dimension in alpha doesn't conflict with 
            # the other parameters.
            tfd.Beta(np.ones(K - 1, dtype), alpha[..., tf.newaxis]),
            reinterpreted_batch_ndims=1
        ),
        # Observations (likelihood)
        obs = lambda mu, sigma, v: tfd.Sample(tfd.MixtureSameFamily(
            # This will be marginalized over.
            mixture_distribution=tfd.Categorical(probs=stickbreak(v)),
            components_distribution=tfd.Normal(mu, sigma)),
            sample_shape=nobs)
    ))

# Example usages:
# dp_sb_gmm = create_dp_sb_gmm(13, 5)
# sample = dp_sb_gmm.sample()
# dp_sb_gmm.log_prob(**sample)
# print(dp_sb_gmm.resolve_graph())
# print(sample)
#
# dp_sb_gmm.log_prob(mu=tfd.Normal(np.float64(0), 1).sample(5),
#                    sigma=tfd.Uniform(np.float64(0), 1).sample(5),
#                    alpha=tf.cast(1, dtype),
#                    v=tfd.Beta(np.float64(1), 1).sample(5 - 1),
#                    obs=np.random.randn(1000))


# In[6]:


# Make sure that `model.sample()` AND `model.sample(N)` works for N > 1!
# create_dp_sb_gmm(13, 5).sample()
# create_dp_sb_gmm(13, 5).sample((1,2))


# In[7]:


# Read simulated data.
path_to_data = '../../data/sim-data/gmm-data-n200.json'
with open(path_to_data) as f:
    simdata = json.load(f)

# Give data the correct type.
y = np.array(simdata['y'], dtype=np.float64)

# Plot histogram of data.
plt.hist(y, density=True, bins=30)
plt.xlabel('data (y)')
plt.ylabel('density')
plt.title('Histogram of data');


# In[8]:


# Helper for plotting posterior distribution of a given parameter.
def plot_param_post(param, param_name, param_full_name, level=95, figsize=(12, 4), truth=None):
    plt.figure(figsize=figsize)
    
    ci_lower = (100 - level) / 2
    ci_upper = (100 + level) / 2

    plt.subplot(1, 2, 1)
    plt.boxplot(param, whis=[ci_lower, ci_upper], showmeans=True, showfliers=False)
    plt.xlabel('mixture components')
    plt.ylabel(param_full_name)
    plt.title('95% Credible Intervals for {}'.format(param_full_name))
    if truth is not None:
        for line in truth:
            plt.axhline(line, ls=':')

    plt.subplot(1, 2, 2)
    plt.plot(param);
    plt.xlabel('iterations')
    plt.ylabel(param_full_name)
    plt.title('Trace plot of {}'.format(param_full_name));


# In[9]:


# Helper for plotting posterior distribution of all model parameters.
def plot_all_params(output, target_logprob_fn):
    mu = output['mu'].numpy()
    sigma = output['sigma'].numpy()
    v = output['v']
    alpha = output['alpha'].numpy()
    eta = stickbreak(v).numpy()
    
    plot_param_post(eta, 'eta', 'mixture weights', truth=simdata['w'])
    plot_param_post(mu, 'mu', 'mixture locations', truth=simdata['mu'])
    plot_param_post(sigma, 'sigma', 'mixture scales', truth=simdata['sig'])
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(alpha, bins=30, density=True);
    plt.xlabel("alpha")
    plt.ylabel("density")
    plt.title("Posterior distribution of alpha"); 
    
    plt.subplot(1, 2, 2)
    # Plot log joint posterior (unnormalized)
    lp = [target_logprob_fn(mu=mu[i], sigma=sigma[i], alpha=alpha[i], v=v[i]) for i in range(len(mu))]
    lp = np.vstack(lp).ravel()
    plt.plot(lp)
    plt.xlabel("iteration (post-burn)")
    plt.ylabel("log joint posterior density (unnormalized)");


# # Model Creation

# In[10]:


# Number of mixture components.
ncomponents = 10

print('Create model ...')
model = create_dp_sb_gmm(nobs=len(simdata['y']), K=ncomponents)

# This allows the model to figure out dimensions of prob vector. 
_ = model.sample()

print('Define log unnormalized joint posterior density ...')
def target_log_prob_fn(mu, sigma, alpha, v):
    return model.log_prob(obs=y, mu=mu, sigma=sigma, alpha=alpha, v=v)


# ***
# # ADVI
# 
# Credits: Thanks to Dave Moore at bayesflow for helping with the implementation!

# In[11]:


# This cell contains everything for initializing the 
# variational distribution, which approximates the true posterior.
# ADVI is quite sensitive to initial distritbution.
tf.random.set_seed(7) # 7

# Create variational parameters.
qmu_loc = tf.Variable(tf.random.normal([ncomponents], dtype=np.float64) * 3, name='qmu_loc')
qmu_rho = tf.Variable(tf.random.normal([ncomponents], dtype=np.float64) * 2, name='qmu_rho')

qsigma_loc = tf.Variable(tf.random.normal([ncomponents], dtype=np.float64) - 2, name='qsigma_loc')
qsigma_rho = tf.Variable(tf.random.normal([ncomponents], dtype=np.float64) - 2, name='qsigma_rho')

qv_loc = tf.Variable(tf.random.normal([ncomponents - 1], dtype=np.float64) - 2, name='qv_loc')
qv_rho = tf.Variable(tf.random.normal([ncomponents - 1], dtype=np.float64) - 1, name='qv_rho')

qalpha_loc = tf.Variable(tf.random.normal([], dtype=np.float64), name='qalpha_loc')
qalpha_rho = tf.Variable(tf.random.normal([], dtype=np.float64), name='qalpha_rho')

# Create variational distribution.
surrogate_posterior = tfd.JointDistributionNamed(dict(
    # qmu
    mu=tfd.Independent(tfd.Normal(qmu_loc, tf.nn.softplus(qmu_rho)), reinterpreted_batch_ndims=1),
    # qsigma
    sigma=tfd.Independent(tfd.LogNormal(qsigma_loc, tf.nn.softplus(qsigma_rho)), reinterpreted_batch_ndims=1),
    # qv
    v=tfd.Independent(tfd.LogitNormal(qv_loc, tf.nn.softplus(qv_rho)), reinterpreted_batch_ndims=1),
    # qalpha
    alpha=tfd.LogNormal(qalpha_loc, tf.nn.softplus(qalpha_rho))))


# In[12]:


# Run optimizer
# @tf.function(autograph=False) , experimental_compile=True)  # Makes slower?
def run_advi(optimizer, sample_size=1, num_steps=2000, seed=1):
    return tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=target_log_prob_fn,
        surrogate_posterior=surrogate_posterior,
        optimizer=optimizer,
        sample_size=sample_size,
        seed=seed, num_steps=num_steps)  # 200, 2000

opt = tf.optimizers.Adam(learning_rate=1e-2)
get_ipython().run_line_magic('time', 'losses = run_advi(opt, sample_size=1)')


# In[13]:


plt.plot(losses.numpy())
plt.xlabel('Optimizer Iteration')
plt.ylabel('ELBO');


# In[14]:


# Extract posterior samples from VI
advi_output = surrogate_posterior.sample(500)
plot_all_params(advi_output, target_log_prob_fn)


# ***
# # MCMC

# In[15]:


# Create initial values..
# For HMC, NUTS. Not necessary for ADVI, as ADVI has surrogate_posterior.
def generate_initial_state():
    return [
        tf.zeros(ncomponents, dtype, name='mu'),
        tf.ones(ncomponents, dtype, name='sigma') * 0.1,
        tf.ones([], dtype, name='alpha') * 0.5,
        tf.fill(ncomponents - 1, value=np.float64(0.5), name='v')
    ]

# Create bijectors to transform unconstrained to and from constrained parameters-space.
# For example, if X ~ Exponential(theta), then X is constrained to be positive. A transformation
# that puts X onto an unconstrained space is Y = log(X). In that case, the bijector used
# should be the **inverse-transform**, which is exp(.) (i.e. so that X = exp(Y)).
#
# NOTE: Define the inverse-transforms for each parameter in sequence.
bijectors = [
    tfb.Identity(),  # mu
    tfb.Exp(),  # sigma
    tfb.Exp(),  # alpha
    tfb.Sigmoid()  # v
]


# In[16]:


# Define HMC sampler.

@tf.function(autograph=False, experimental_compile=True)
def hmc_sample(num_results, num_burnin_steps, current_state, step_size=0.01, num_leapfrog_steps=100):
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=current_state,
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=target_log_prob_fn,
                    step_size=step_size, num_leapfrog_steps=num_leapfrog_steps, seed=1),
                bijector=bijectors),
            num_adaptation_steps=num_burnin_steps),
        trace_fn = lambda _, pkr: pkr.inner_results.inner_results.is_accepted)


# ## HMC

# In[17]:


tf.random.set_seed(7)

# Compile time.
current_state = generate_initial_state()  # generate initial values.
get_ipython().run_line_magic('time', '[mu, sigma, alpha, v], is_accepted = hmc_sample(1, 1, current_state=current_state)')

# Run time.
current_state = generate_initial_state()  # generate initial values.
get_ipython().run_line_magic('time', '[mu, sigma, alpha, v], is_accepted = hmc_sample(500, 500, current_state=current_state)  # 14.2 seconds.')

# Store posterior samples.
hmc_output = dict(mu=mu, sigma=sigma, alpha=alpha, v=v, acceptance_rate=is_accepted.numpy().mean())


# In[18]:


# HMC posterior inference
plot_all_params(hmc_output, target_log_prob_fn)


# ## NUTS

# In[19]:


# Define Nuts sampler.

# Improve performance by tracing the sampler using `tf.function`
# and compiling it using XLA.
@tf.function(autograph=False, experimental_compile=True)
def nuts_sample(num_results, num_burnin_steps, current_state, max_tree_depth=10):
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=current_state,
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=tfp.mcmc.NoUTurnSampler(
                     target_log_prob_fn=target_log_prob_fn,
                     max_tree_depth=max_tree_depth, step_size=0.01, seed=1),
                bijector=bijectors),
            num_adaptation_steps=num_burnin_steps,  # should be smaller than burn-in.
            target_accept_prob=0.8),
        trace_fn = lambda _, pkr: pkr.inner_results.inner_results.is_accepted)


# In[20]:


tf.random.set_seed(7)

# Compile time.
current_state = generate_initial_state()
get_ipython().run_line_magic('time', '[mu, sigma, alpha, v], is_accepted = nuts_sample(1, 1, current_state=current_state, max_tree_depth=1)  # 15 seconds.')

# Run time.
current_state = generate_initial_state()
get_ipython().run_line_magic('time', '[mu, sigma, alpha, v], is_accepted = nuts_sample(500, 500, current_state=current_state)  # 36 seconds.')

# Store posterior samples.
nuts_output = dict(mu=mu, sigma=sigma, alpha=alpha, v=v, acceptance_rate=is_accepted.numpy().mean())


# In[21]:


# NUTS posterior inference
plot_all_params(nuts_output, target_log_prob_fn)


# In[ ]:




