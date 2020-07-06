import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed


def stickbreak(v):
    cumprod_one_minus_v = tf.math.cumprod(1 - v)
    one_v = tf.pad(v, [[0, 1]], "CONSTANT", constant_values=1)
    c_one = tf.pad(cumprod_one_minus_v, [[1, 0]], "CONSTANT", constant_values=1)
    return one_v * c_one 


# NOTE: Not sure how to do this. Insufficient documentation ...
# def dp_gmm_sb(y, K):
#     mu = ed.Normal(tf.zeros(K), 1.0, name="mu")
#     sigma = ed.Normal(tf.ones(K), rate=10, name="sigma")
#     alpha = ed.Gamma(1, rate=10, name="alpha")
#     v = ed.Beta(tf.ones(K - 1), alpha, name="v")
# 
#     obs = ed.MixtureSameFamily(mixture_distributions=ed.Normal(mu, sigma),
#                                components_distribution=ed.Categorial(stickbreak(eta)),
#                                name='obs')
# 
#     return obs
