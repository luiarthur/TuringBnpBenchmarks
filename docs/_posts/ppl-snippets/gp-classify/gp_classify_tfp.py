# Import libraries.
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from sklearn.datasets import make_moons
from tqdm import trange

sigmoid = lambda x: 1 / (1 + np.exp(-x))

# Alias.
SqExpKernel = tfp.math.psd_kernels.ExponentiatedQuadratic

# Default data type for tensorflow tensors.
dtype = np.float64

# Set random seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

# Read data
X, y = make_moons(n_samples=50, shuffle=True, noise=0.1, random_state=1)

# Store data.
X = np.stack([simdata['x1'], simdata['x2']], axis=-1).astype(dtype)
y = np.array(simdata['y'])


# Specify GP model.
def compute_LK(alpha, rho, X, jitter=1e-6):
    kernel = SqExpKernel(alpha, rho)
    K = kernel.matrix(X, X) + tf.eye(X.shape[0], dtype=dtype) * jitter
    return tf.linalg.cholesky(K)

def compute_f(alpha, rho, beta, eta):
    LK = compute_LK(alpha, rho, X)
    f = tf.linalg.matvec(LK, eta)  # LK * eta, (matrix * vector)
    return f + beta[..., tf.newaxis]

# GP Binary Classification Model.
gpc_model = tfd.JointDistributionNamed(dict(
    alpha=tfd.LogNormal(dtype(0), dtype(1)),
    rho=tfd.LogNormal(dtype(0), dtype(1)),
    beta=tfd.Normal(dtype(0), dtype(1)),
    eta=tfd.Sample(tfd.Normal(dtype(0), dtype(1)),
                   sample_shape=X.shape[0]),
    # NOTE: `Sample` and `Independent` resemble, respectively,
    # `filldist` and `arraydist` in Turing.
    obs=lambda alpha, rho, beta, eta: tfd.Independent(
        tfd.Bernoulli(logits=compute_f(alpha, rho, beta, eta)),
        reinterpreted_batch_ndims=1) 
))


### MODEL SET UP ###

# For some reason, this is needed for the compiler
# to know the correct model parameter dimensions.
_ = gpc_model.sample()


# Parameters as they appear in model definition.
# NOTE: Initial values should be defined in order appeared in model.
ordered_params = ['alpha', 'rho', 'beta', 'eta']

# Initial values.
tf.random.set_seed(1)
s = gpc_model.sample()
initial_state = [s[key] for key in ordered_params]

# Bijectors (from unconstrained to constrained space)
bijectors = [
    tfp.bijectors.Exp(),  # alpha
    tfp.bijectors.Exp(),  # rho
    tfp.bijectors.Identity(),  # beta
    tfp.bijectors.Identity(),  # eta
]

# Unnormalized log posterior
def unnormalized_log_posterior(alpha, rho, beta, eta):
    return gpc_model.log_prob(alpha=alpha, rho=rho,
                              beta=beta, eta=eta, obs=y)


### HMC ###
def hmc_trace_fn(state, pkr):
    """
    state: current state in MCMC
    pkr: previous kernel result
    """
    result = pkr.inner_results
    return dict(is_accepted=result.is_accepted,
                target_log_prob=result.accepted_results.target_log_prob)
    
@tfp.experimental.nn.util.tfcompile
def run_hmc(num_results, num_burnin_steps):
      return tfp.mcmc.sample_chain(
          num_results=num_results,
          num_burnin_steps=num_burnin_steps,
          current_state=initial_state,
          kernel=tfp.mcmc.TransformedTransitionKernel(
                     inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                         target_log_prob_fn=unnormalized_log_posterior,
                         step_size=0.05,
                         num_leapfrog_steps=20),
                     bijector=bijectors),
          trace_fn=hmc_trace_fn)

# set random seed
tf.random.set_seed(2)
# Run HMC sampler
[alpha, rho, beta, eta], hmc_stats = run_hmc(500, 500)


### NUTS ###
def nuts_trace_fn(state, pkr):
    """
    state: current state in MCMC
    pkr: previous kernel result
    """
    result = pkr.inner_results.inner_results
    return dict(is_accepted=result.is_accepted,
                target_log_prob=result.target_log_prob)
  
@tfp.experimental.nn.util.tfcompile
def run_nuts(num_results, num_burnin_steps):
      return tfp.mcmc.sample_chain(
          num_results=num_results,
          num_burnin_steps=num_burnin_steps,
          current_state=initial_state,
          kernel=tfp.mcmc.SimpleStepSizeAdaptation(
              tfp.mcmc.TransformedTransitionKernel(
                  inner_kernel = tfp.mcmc.NoUTurnSampler(
                      target_log_prob_fn=unnormalized_log_posterior,
                      max_tree_depth=10, step_size=0.05),
                  bijector=bijectors),
          num_adaptation_steps=num_burnin_steps,
          target_accept_prob=0.8),
          trace_fn=nuts_trace_fn)

# set random seed
tf.random.set_seed(2)
# Run NUTS sampler
[alpha, rho, beta, eta], nuts_stats = run_nuts(500, 500)


### ADVI ###
tf.random.set_seed(3)

# Create variational parameters.
vp_dict = dict()
for key in ordered_params:
    param_shape = gpc_model.model[key].event_shape
    vp_dict[f'q{key}_loc'] = tf.Variable(
        tf.random.normal(param_shape, dtype=dtype), name=f'q{key}_loc')
    vp_dict[f'q{key}_rho'] = tf.Variable(
        tf.random.normal(param_shape, dtype=dtype), name=f'q{key}_rho')
    
# Create variational distribution.
surrogate_family = dict(alpha=tfd.LogNormal, rho=tfd.LogNormal,
                        beta=tfd.Normal, eta=tfd.Normal)
surrogate_posterior_dict = {
    key: surrogate_family[key](vp_dict[f'q{key}_loc'],
                               tf.nn.softplus(vp_dict[f'q{key}_rho']))
    for key in ordered_params
}
surrogate_posterior_dict['eta'] = tfd.Independent(
    surrogate_posterior_dict['eta'], reinterpreted_batch_ndims=1)
surrogate_posterior = tfd.JointDistributionNamed(
    surrogate_posterior_dict
)
    
# Function for running ADVI.
def run_advi(sample_size, num_steps):
    return tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=unnormalized_log_posterior,
        surrogate_posterior=surrogate_posterior,
        optimizer=tf.optimizers.Adam(learning_rate=1e-1),
        seed=1,
        sample_size=sample_size,  # ELBO samples.
        num_steps=num_steps)  # Number of iterations to run optimizer.

# Fit GP via ADVI.
losses = run_advi(sample_size=10, num_steps=1000)

# Extract posterior samples from variational distributions.
advi_samples = surrogate_posterior.sample(500)
advi_samples = {k: advi_samples[k].numpy() for k in advi_samples}
