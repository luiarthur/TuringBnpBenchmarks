# NOTE: Import libraries...

# NOTE: Read data ...

# Default data type for tensorflow tensors.
dtype = np.float64

# Set random seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

# Here we will use the squared exponential covariance function:
# 
# $$
# \alpha^2 \cdot \exp\left\{-\frac{d^2}{2\rho^2}\right\}
# $$
# 
# where $\alpha$ is the amplitude of the covariance, $\rho$ is the length scale
# which controls how slowly information decays with distance (larger $\rho$
# means information about a point can be used for data far away); and $d$ is
# the distance.

# Specify GP model
gp_model = tfd.JointDistributionNamed(dict(
    amplitude=tfd.LogNormal(dtype(0), dtype(1)),  # amplitude
    length_scale=tfd.LogNormal(dtype(-2), dtype(0.1)),  # length scale
    obs=lambda length_scale, amplitude: tfd.GaussianProcess(
          kernel=tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude, length_scale),
          index_points=X[..., np.newaxis],
          observation_noise_variance=dtype(0), jitter=1e-3)
))

# Run graph to make sure it works.
_ = gp_model.sample()

# Initial values.
initial_state = [
    1e-1 * tf.ones([], dtype=np.float64, name='amplitude'),
    1e-1 * tf.ones([], dtype=np.float64, name='length_scale')
]

# Bijectors (from unconstrained to constrained space)
bijectors = [
    tfp.bijectors.Softplus(),  # amplitude
    tfp.bijectors.Softplus()  # length_scale
]

# Unnormalized log posterior
def unnormalized_log_posterior(amplitude, length_scale):
    return gp_model.log_prob(amplitude=amplitude, length_scale=length_scale, obs=y)

# Create a function to run HMC.
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


# Run HMC.
[amplitudes, length_scales], is_accepted = run_hmc(1000, 1000)


# Create function to run NUTS.
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


# Run NUTS.
[amplitudes, length_scales], is_accepted = run_nuts(1000, 1000)


### ADVI ###
# Create variational parameters.
qamp_loc = tf.Variable(tf.random.normal([], dtype=dtype) - 1, name='qamp_loc')
qamp_rho = tf.Variable(tf.random.normal([], dtype=dtype) - 1, name='qamp_rho')

qlength_loc = tf.Variable(tf.random.normal([], dtype=dtype), name='qlength_loc')
qlength_rho = tf.Variable(tf.random.normal([], dtype=dtype), name='qlength_rho')

# Create variational distribution.
surrogate_posterior = tfd.JointDistributionNamed(dict(
    amplitude=tfd.LogNormal(qamp_loc, tf.nn.softplus(qamp_rho)),
    length_scale=tfd.LogNormal(qlength_loc, tf.nn.softplus(qlength_rho))
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

# Fit GP via ADVI.
losses = run_advi(sample_size=1, num_steps=2000)
