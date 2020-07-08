def stickbreak(v):
    cumprod_one_minus_v = tf.math.cumprod(1 - v)
    one_v = tf.pad(v, [[0, 1]], "CONSTANT", constant_values=1)
    c_one = tf.pad(cumprod_one_minus_v, [[1, 0]], "CONSTANT", constant_values=1)
    return one_v * c_one 


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

        # Observations (likelihood)
        obs = lambda mu, sigma, v: tfd.Sample(tfd.MixtureSameFamily(
            # This will be marginalized over.
            mixture_distribution=tfd.Categorical(probs=stickbreak(v)),
            components_distribution=tfd.Normal(mu, sigma)),
            sample_shape=nobs)
    ))


# NOTE: Not able to successfully implement ADVI in TFP due to lack of
# documentation.


# Create Model
ncomponents = 10
model = create_dp_sb_gmm(nobs=len(simdata['y']), K=ncomponents)

# Define log joint density.
def joint_log_prob(obs, mu, sigma, alpha, v):
    return model.log_prob(obs=obs, 
                          mu=mu, sigma=sigma,
                          alpha=alpha, v=v)

unnormalized_posterior_log_prob = functools.partial(joint_log_prob, y)

# Create initial state.
initial_state = [
    tf.zeros(ncomponents, dtype, name='mu'),
    tf.ones(ncomponents, dtype, name='sigma') * .1,
    tf.ones([], dtype, name='alpha'),
    tf.fill(ncomponents - 1, value=np.float64(0.5), name='v')
]

# Create bijectors to transform unconstrained to and from constrained parameters-space.
# For example, if X ~ Exponential(theta), then X is constrained to be positive. A transformation
# that puts X onto an unconstrained space is Y = log(X). In that case, the bijector used
# should be the **inverse-transform**, which is exp(.) (i.e. so that X = exp(Y)).

# Define the inverse-transforms for each parameter in sequence.
bijectors = [
    tfb.Identity(),  # mu
    tfb.Exp(),  # sigma
    tfb.Exp(),  # alpha
    tfb.Sigmoid()  # v
]

   
print('Define sampler ...')
@tf.function(autograph=False)
def sample(use_nuts, max_tree_depth=10):
    if use_nuts:
        ### NUTS ###
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=tfp.mcmc.NoUTurnSampler(
                     target_log_prob_fn=unnormalized_posterior_log_prob,
                     max_tree_depth=max_tree_depth, step_size=0.1, seed=1),
                bijector=bijectors),
            num_adaptation_steps=400,  # should be smaller than burn-in.
            target_accept_prob=0.8)
        trace_fn = lambda _, pkr: pkr.inner_results.inner_results.is_accepted
    else:
        ### HMC ###
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=unnormalized_posterior_log_prob,
                    step_size=0.01, num_leapfrog_steps=100, seed=1),
                bijector=bijectors),
            num_adaptation_steps=400)  # should be smaller than burn-in.
        trace_fn = lambda _, pkr: pkr.inner_results.inner_results.is_accepted

    return tfp.mcmc.sample_chain(
        num_results=500,
        num_burnin_steps=500,
        current_state=initial_state,
        kernel=kernel,
        trace_fn=trace_fn)

# Run HMC sampler
[mu, sigma, alpha, v], is_accepted = sample(use_nuts=False)  # 53 seconds.
hmc_output = dict(mu=mu, sigma=sigma, alpha=alpha, v=v,
                  acceptance_rate=is_accepted.numpy().mean())


# Run NUTS sampler
[mu, sigma, alpha, v], is_accepted = sample(use_nuts=True)  # 9min 15s
nuts_output = dict(mu=mu, sigma=sigma, alpha=alpha, v=v,
                   acceptance_rate=is_accepted.numpy().mean())
