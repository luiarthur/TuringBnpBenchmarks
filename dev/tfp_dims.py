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
m = tfd.Normal(loc=np.ones(3), scale=1.0)
ind = tfd.Independent(m, reinterpreted_batch_ndims=None)
ind.log_prob(np.random.randn(100, ))
