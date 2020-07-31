# import libraries here ...

# Stickbreak function
def stickbreak(v):
    batch_ndims = len(v.shape) - 1
    cumprod_one_minus_v = tf.math.cumprod(1 - v, axis=-1)
    one_v = tf.pad(v, [[0, 0]] * batch_ndims + [[0, 1]], "CONSTANT",
                   constant_values=1)
    c_one = tf.pad(cumprod_one_minus_v, [[0, 0]] * batch_ndims + [[1, 0]],
                   "CONSTANT", constant_values=1)
    return one_v * c_one

# See: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MixtureSameFamily
# See: https://www.tensorflow.org/probability/examples/Bayesian_Gaussian_Mixture_Model
# Define model builder.
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
            # tfd.Beta(np.ones(K - 1, dtype), alpha),
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
            # mixture_distribution=tfd.Categorical(probs=v),
            components_distribution=tfd.Normal(mu, sigma)),
            sample_shape=nobs)
    ))

# Number of mixture components.
ncomponents = 10

# Create model.
model = create_dp_sb_gmm(nobs=len(simdata['y']), K=ncomponents)

# Define log unnormalized joint posterior density.
def target_log_prob_fn(mu, sigma, alpha, v):
    return model.log_prob(obs=y, mu=mu, sigma=sigma, alpha=alpha, v=v)

# NOTE: Read data y here ...
# Here, y (a vector of length 500) is noisy univariate draws from a
# mixture distribution with 4 components.

### ADVI ###
# Prep work for ADVI. Credit: Thanks to Dave Moore at BayesFlow for helping
# with the implementation!

# ADVI is quite sensitive to initial distritbution.
tf.random.set_seed(7) # 7

# Create variational parameters.
qmu_loc = tf.Variable(tf.random.normal([ncomponents], dtype=np.float64) * 3,
                      name='qmu_loc')
qmu_rho = tf.Variable(tf.random.normal([ncomponents], dtype=np.float64) * 2,
                      name='qmu_rho')

qsigma_loc = tf.Variable(tf.random.normal([ncomponents], dtype=np.float64) - 2,
                         name='qsigma_loc')
qsigma_rho = tf.Variable(tf.random.normal([ncomponents], dtype=np.float64) - 2,
                         name='qsigma_rho')

qv_loc = tf.Variable(tf.random.normal([ncomponents - 1], dtype=np.float64) - 2,
                     name='qv_loc')
qv_rho = tf.Variable(tf.random.normal([ncomponents - 1], dtype=np.float64) - 1,
                     name='qv_rho')

qalpha_loc = tf.Variable(tf.random.normal([], dtype=np.float64),
                         name='qalpha_loc')
qalpha_rho = tf.Variable(tf.random.normal([], dtype=np.float64),
                         name='qalpha_rho')

# Create variational distribution.
surrogate_posterior = tfd.JointDistributionNamed(dict(
    # qmu
    mu=tfd.Independent(tfd.Normal(qmu_loc, tf.nn.softplus(qmu_rho)),
                       reinterpreted_batch_ndims=1),
    # qsigma
    sigma=tfd.Independent(tfd.LogNormal(qsigma_loc,
                                        tf.nn.softplus(qsigma_rho)),
                          reinterpreted_batch_ndims=1),
    # qv
    v=tfd.Independent(tfd.LogitNormal(qv_loc, tf.nn.softplus(qv_rho)),
                      reinterpreted_batch_ndims=1),
    # qalpha
    alpha=tfd.LogNormal(qalpha_loc, tf.nn.softplus(qalpha_rho))))

# Run ADVI.
losses = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=target_log_prob_fn,
    surrogate_posterior=surrogate_posterior,
    optimizer=tf.optimizers.Adam(learning_rate=1e-2),
    sample_size=100, seed=1, num_steps=2000)  # 9 seconds


### MCMC (HMC/NUTS) ###

# Creates initial values for HMC, NUTS. 
def generate_initial_state(seed=None):
    tf.random.set_seed(seed)
    return [
        tf.zeros(ncomponents, dtype, name='mu'),
        tf.ones(ncomponents, dtype, name='sigma') * 0.1,
        tf.ones([], dtype, name='alpha') * 0.5,
        tf.fill(ncomponents - 1, value=np.float64(0.5), name='v')
    ]

# Create bijectors to transform unconstrained to and from constrained
# parameters-space.  For example, if X ~ Exponential(theta), then X is
# constrained to be positive. A transformation that puts X onto an
# unconstrained # space is Y = log(X). In that case, the bijector used should
# be the **inverse-transform**, which is exp(.) (i.e. so that X = exp(Y)).
#
# NOTE: Define the inverse-transforms for each parameter in sequence.
bijectors = [
    tfb.Identity(),  # mu
    tfb.Exp(),  # sigma
    tfb.Exp(),  # alpha
    tfb.Sigmoid()  # v
]

# Define HMC sampler.
@tf.function(autograph=False, experimental_compile=True)
def hmc_sample(num_results, num_burnin_steps, current_state, step_size=0.01,
               num_leapfrog_steps=100):
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


# Define NUTS sampler.
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

# Run HMC sampler.
current_state = generate_initial_state()
[mu, sigma, alpha, v], is_accepted = hmc_sample(500, 500, current_state)
hmc_output = dict(mu=mu, sigma=sigma, alpha=alpha, v=v,
                  acceptance_rate=is_accepted.numpy().mean())

# Run NUTS sampler.
current_state = generate_initial_state()
[mu, sigma, alpha, v], is_accepted = nuts_sample(500, 500, current_state)
nuts_output = dict(mu=mu, sigma=sigma, alpha=alpha, v=v,
                   acceptance_rate=is_accepted.numpy().mean())
