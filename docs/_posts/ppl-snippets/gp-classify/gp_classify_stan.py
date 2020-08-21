# Load libraries.
import json
import numpy as np
import matplotlib.pyplot as plt
import pystan
from tqdm import trange
from scipy.spatial import distance_matrix
from scipy.stats import norm
from sklearn.datasets import make_moons
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF


# Create data dictionary.
def create_stan_data(X, y, m_rho=0, s_rho=1, m_alpha=0, s_alpha=1, eps=1e-6):
    N, P = X.shape
    assert (N, ) == y.shape
    
    return dict(y=y, X=X, N=N, P=P,
                m_rho=m_rho, s_rho=s_rho,
                m_alpha=m_alpha, s_alpha=s_alpha, eps=eps)


# GP binary classification STAN model code.
model_code = """
data {
  int<lower=0> N;
  int<lower=0> P;
  int<lower=0, upper=1> y[N];
  matrix[N, P] X;
  real m_rho;
  real<lower=0> s_rho;
  real m_alpha;
  real<lower=0> s_alpha;
  real<lower=0> eps;
}

parameters {
  real<lower=0> rho;   // range parameter in GP covariance fn
  real<lower=0> alpha; // covariance scale parameter in GP covariance fn
  vector[N] eta;
  real beta;
}

transformed parameters {
  vector[N] f;
  {
    matrix[N, N] K;   // GP covariance matrix
    matrix[N, N] LK;  // cholesky of GP covariance matrix
    row_vector[N] row_x[N];
    
    // Using exponential quadratic covariance function
    for (n in 1:N) {
      row_x[n] = to_row_vector(X[n, :]);
    }
    K = cov_exp_quad(row_x, alpha, rho); 

    // Add small values along diagonal elements for numerical stability.
    for (n in 1:N) {
        K[n, n] = K[n, n] + eps;
    }

    // Cholesky of K (lower triangle).  
    LK = cholesky_decompose(K); 
  
    f = LK * eta;
  }
}

model {
  // Priors.
  alpha ~ lognormal(m_alpha, s_alpha);
  rho ~ lognormal(m_rho, s_rho);
  eta ~ std_normal();
  beta ~ std_normal();
 
  // Model.
  y ~ bernoulli_logit(beta + f);
}
"""


# Compile model. This takes about a minute.
sm = pystan.StanModel(model_code=model_code)


# Make data.
X, y = make_moons(n_samples=50, shuffle=True, noise=0.1, random_state=1)

# Generate stan data dictionary.
stan_data = create_stan_data(X, y)

# Fit via ADVI.
vb_fit = sm.vb(data=stan_data, iter=1000, seed=1,
               grad_samples=1, elbo_samples=1)

# Fit via HMC.
# - stepsize = 0.05
# - num leapfrog steps = 20
# - burn in: 500
# - samples: 500
hmc_fit = sm.sampling(
    data=stan_data, iter=1000, warmup=500, thin=1,
    seed=1, algorithm='HMC', chains=1,
    control=dict(stepsize=0.05, int_time=1, adapt_engaged=False))


# Fit via NUTS.
nuts_fit = sm.sampling(data=stan_data, iter=1000, warmup=500, thin=1,
                       seed=1, algorithm='NUTS', chains=1)
