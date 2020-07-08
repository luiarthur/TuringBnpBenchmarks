model = """
data {
  int<lower=0> K;  // Number of cluster
  int<lower=0> N;  // Number of observations
  real y[N];  // observations
  real<lower=0> alpha_shape;
  real<lower=0> alpha_rate;
  real<lower=0> sigma_shape;
  real<lower=0> sigma_rate;
}

parameters {
  real mu[K]; // cluster means
  // real <lower=0,upper=1> v[K - 1];  // stickbreak components
  vector<lower=0,upper=1>[K - 1] v;  // stickbreak components
  real<lower=0> sigma[K];  // error scale
  real<lower=0> alpha;  // hyper prior DP(alpha, base)
}

transformed parameters {
  simplex[K] eta;
  vector<lower=0,upper=1>[K - 1] cumprod_one_minus_v;

  cumprod_one_minus_v = exp(cumulative_sum(log1m(v)));
  eta[1] = v[1];
  eta[2:(K-1)] = v[2:(K-1)] .* cumprod_one_minus_v[1:(K-2)];
  eta[K] = cumprod_one_minus_v[K - 1];
}

model {
  real ps[K];
  // real alpha = 1;
  
  alpha ~ gamma(alpha_shape, alpha_rate);  // mean = a/b = shape/rate 
  sigma ~ gamma(sigma_shape, sigma_rate);
  mu ~ normal(0, 3);
  v ~ beta(1, alpha);

  for(i in 1:N){
    for(k in 1:K){
      ps[k] = log(eta[k]) + normal_lpdf(y[i] | mu[k], sigma[k]);
    }
    target += log_sum_exp(ps);
  }
}

generated quantities {
  real ll;
  real ps_[K];
  
  ll = 0;
  for(i in 1:N){
    for(k in 1:K){
      ps_[k] = log(eta[k]) + normal_lpdf(y[i] | mu[k], sigma[k]);
    }
    ll += log_sum_exp(ps_);
  }  
}
"""


# Compile the model.
get_ipython().run_line_magic('time', 'sm = pystan.StanModel(model_code=model)')



# Approximate posterior via ADVI
# - ADVI is sensitive to starting values. Should run several times and pick run 
#   that has best fit (e.g. highest ELBO / logliklihood).
# - Variational inference works better with more data. Inference is less accurate
#   with small datasets, due to the variational approximation.

fit = sm.vb(data=data, iter=1000, seed=1, algorithm='meanfield',
            adapt_iter=1000, verbose=False, grad_samples=1, elbo_samples=100,
            adapt_engaged=True, output_samples=1000)


# MCMC setup

# Number of burn in iterations
burn = 500

# Number of sampels to keep
nsamples = 500

# Number of MCMC (HMC / NUTS) iterations in total
niters = burn + nsamples


# Sample from posterior via HMC
# NOTE: num_leapfrog = int_time / stepsize.
hmc_fit = sm.sampling(data=data, iter=niters, chains=1, warmup=burn, thin=1,
                      seed=1, algorithm='HMC',
                      control=dict(stepsize=0.01, int_time=1))



# Sample from posterior via NUTS
nuts_fit = sm.sampling(data=data, iter=niters, chains=1, warmup=burn, thin=1,
                       seed=1)

