import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# sample_n_shape = [n] + batch_shape + event_shape, where
# - sample_n_shape is the shape of the Tensor returned from sample(n), n
#   is the number of samples,
# - batch_shape defines how many independent distributions there are, and
# - event_shape defines the shape of samples from each of those 
#   independent distributions.
# Samples are independent along the batch_shape dimensions, but
# not necessarily so along the event_shape dimensions (depending on the
# particulars of the underlying distribution). 


# Create 6 independent Normal distributions, represented in a 3x2 Normal
# distribution object.
x = tfd.Normal(loc=np.ones((3, 2)), scale=1.0)

# View the 3x2 independnet Normals (of scalar shape) in one big Independent
# object which has shape 3x2.
ind = tfd.Independent(x, reinterpreted_batch_ndims=2)

# Compute log prob
ind.log_prob(np.random.randn(100, 3, 2))

# GMM
x = np.random.randn(2)
m = tfd.Normal(loc=np.ones((1, 3)), scale=1.0)
ind = tfd.Independent(m)
ind.log_prob(x[:, None])
m.log_prob(x[:, None])

ncomponents = 10
bgmm = tfd.JointDistributionNamed(dict(
  mix_probs=tfd.Dirichlet(
    concentration=np.ones(components, dtype) / 10.),
  loc=tfd.Independent(
    tfd.Normal(
        loc=np.stack([
            -np.ones(dims, dtype),
            np.zeros(dims, dtype),
            np.ones(dims, dtype),
        ]),
        scale=tf.ones([components, dims], dtype)),
    reinterpreted_batch_ndims=2),
  precision=tfd.Independent(
    tfd.WishartTriL(
        df=5,
        scale_tril=np.stack([np.eye(dims, dtype=dtype)]*components),
        input_output_cholesky=True),
    reinterpreted_batch_ndims=1),
  s=lambda mix_probs, loc, precision: tfd.Sample(tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(probs=mix_probs),
      components_distribution=MVNCholPrecisionTriL(
          loc=loc,
          chol_precision_tril=precision)),
      sample_shape=num_samples)
))
