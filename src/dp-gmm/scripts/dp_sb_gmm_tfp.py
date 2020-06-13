#!/usr/bin/env python
# coding: utf-8

# In[16]:


# TODO: Clean up this implementation!


# In[1]:


import functools
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
dtype = tf.float64

# Resources for learning TFP
# https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/NoUTurnSampler
# https://www.tensorflow.org/probability/overview
# https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc
# https://www.tensorflow.org/probability/examples/Modeling_with_JointDistribution
# https://www.tensorflow.org/probability/examples/Bayesian_Gaussian_Mixture_Model

# To better understand event_shape, batch_shape, sample_shape:
# https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution
# https://www.youtube.com/watch?v=zWXTpZX4PPo


# In[2]:


# Set random seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)


# In[3]:


def stickbreak(v):
    cumprod_one_minus_v = tf.math.cumprod(1 - v)
    one_v = tf.pad(v, [[0, 1]], "CONSTANT", constant_values=1)
    c_one = tf.pad(cumprod_one_minus_v, [[1, 0]], "CONSTANT", constant_values=1)
    return one_v * c_one 

# Example:
# stickbreak(np.random.rand(3))


# In[4]:


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
            tfd.Gamma(concentration=np.ones(K, dtype), rate=10),
            reinterpreted_batch_ndims=1
        ),
        # Mixture weights (stick-breaking construction)
        alpha = tfd.Gamma(concentration=np.float64(1.0), rate=10.0),
        v = lambda alpha: tfd.Independent(
            tfd.Beta(np.ones(K - 1, dtype), alpha),
            reinterpreted_batch_ndims=1
        ),
        # v = tfd.Dirichlet(np.ones(K, dtype) / K),

        # Observations (likelihood)
        obs = lambda mu, sigma, v: tfd.Sample(tfd.MixtureSameFamily(
            # This will be marginalized over.
            mixture_distribution=tfd.Categorical(probs=stickbreak(v)),
            # mixture_distribution=tfd.Categorical(probs=v),
            components_distribution=tfd.Normal(mu, sigma)),
            sample_shape=nobs)
    ))


# In[5]:


# print(dp_sb_gmm.resolve_graph())
dp_sb_gmm = create_dp_sb_gmm(13, 5)
sample = dp_sb_gmm.sample()
dp_sb_gmm.log_prob(**sample)
# sample


# In[6]:


sample


# In[7]:


dp_sb_gmm.log_prob(mu=tfd.Normal(np.float64(0), 1).sample(5),
                   sigma=tfd.Uniform(np.float64(0), 1).sample(5),
                   alpha=tf.cast(1, dtype),
                   v=tfd.Beta(np.float64(1), 1).sample(5 - 1),
                   obs=np.random.randn(1000))


# In[8]:


# Read simulated data.
path_to_data = '../../data/sim-data/gmm-data-n200.json'
with open(path_to_data) as f:
    simdata = json.load(f)
    
y = np.array(simdata['y'], np.float64)


# In[38]:


get_ipython().run_cell_magic('time', '', "\nncomponents = 10\n\nprint('Create model ...')\nmodel = create_dp_sb_gmm(nobs=len(simdata['y']), K=ncomponents)\n\nprint('Define log joint density ...')\ndef joint_log_prob(obs, mu, sigma, alpha, v):\n    return model.log_prob(obs=obs, \n                          mu=mu, sigma=sigma,\n                          alpha=alpha, v=v)\n    \n# Test log joint density evaluation\n# _ = joint_log_prob(y, \n#                    np.random.randn(10),\n#                    np.random.rand(10),\n#                    np.float64(1),\n#                    np.random.rand(9))\n\n# I don't know why this is necessary...\nunnormalized_posterior_log_prob = functools.partial(joint_log_prob, y)\n\n# Create initial state?\ninitial_state = [\n    tf.zeros(ncomponents, dtype, name='mu'),\n    tf.ones(ncomponents, dtype, name='sigma') * .1,\n    tf.ones([], dtype, name='alpha'),\n    tf.fill(ncomponents - 1, value=np.float64(0.5), name='v')\n]\n\n# Create bijectors to transform unconstrained to and from constrained parameters-space.\n# For example, if X ~ Exponential(theta), then X is constrained to be positive. A transformation\n# that puts X onto an unconstrained space is Y = log(X). In that case, the bijector used\n# should be the **inverse-transform**, which is exp(.) (i.e. so that X = exp(Y)).\n\n# Define the inverse-transforms for each parameter in sequence.\nbijectors = [\n    tfb.Identity(),  # mu\n    tfb.Exp(),  # sigma\n    tfb.Exp(),  # alpha\n    tfb.Sigmoid()  # v\n]\n\n   \nprint('Define sampler ...')\n@tf.function(autograph=False)\ndef sample(use_nuts, max_tree_depth=10):\n    if use_nuts:\n        ### NUTS ###\n        kernel = tfp.mcmc.SimpleStepSizeAdaptation(\n            tfp.mcmc.TransformedTransitionKernel(\n                inner_kernel=tfp.mcmc.NoUTurnSampler(\n                     target_log_prob_fn=unnormalized_posterior_log_prob,\n                     max_tree_depth=max_tree_depth, step_size=0.1, seed=1),\n                bijector=bijectors),\n            num_adaptation_steps=400,  # should be smaller than burn-in.\n            target_accept_prob=0.8)\n        trace_fn = lambda _, pkr: pkr.inner_results.inner_results.is_accepted\n    else:\n        ### HMC ###\n        kernel = tfp.mcmc.SimpleStepSizeAdaptation(\n            tfp.mcmc.TransformedTransitionKernel(\n                inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(\n                    target_log_prob_fn=unnormalized_posterior_log_prob,\n                    step_size=0.01, num_leapfrog_steps=100, seed=1),\n                bijector=bijectors),\n            num_adaptation_steps=400)  # should be smaller than burn-in.\n        trace_fn = lambda _, pkr: pkr.inner_results.inner_results.is_accepted\n\n    return tfp.mcmc.sample_chain(\n        num_results=500,\n        num_burnin_steps=500,\n        current_state=initial_state,\n        kernel=kernel,\n        trace_fn=trace_fn)\n\nprint('Run sampler ...')\n# %time [mu, sigma, alpha, v], is_accepted = sample(use_nuts=False)  # 53 seconds.\n# %time [mu, sigma, alpha, v], is_accepted = sample(use_nuts=True, max_tree_depth=5)  # 40.9 seconds\n%time [mu, sigma, alpha, v], is_accepted = sample(use_nuts=True)  # 9min 15s")


# In[39]:


is_accepted.numpy().mean()


# In[67]:


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


# In[68]:


eta = np.apply_along_axis(stickbreak, 1, v)
plot_param_post(eta, 'eta', 'mixture weights (eta)', truth=simdata['w']);


# In[73]:


plot_param_post(mu.numpy(), 'mu', 'mixture means (mu)', truth=simdata['mu']);


# In[74]:


plot_param_post(sigma.numpy(), 'sigma', 'mixture scales (sigma)', truth=simdata['sig']);


# In[46]:


plt.hist(alpha.numpy(), density=True, bins=30)
plt.ylabel('density')
plt.xlabel('alpha');


# In[88]:


lp = [joint_log_prob(y, mu[i], sigma[i], alpha[i], v[i]) for i in range(len(mu))]
lp = np.vstack(lp).ravel()


# In[90]:


# Plot likelihood
plt.plot(lp)
plt.xlabel("iteration (post-burn)")
plt.ylabel("log likelihood");


# In[ ]:




