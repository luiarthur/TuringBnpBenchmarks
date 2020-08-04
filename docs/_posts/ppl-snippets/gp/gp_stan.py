# Import libraries ...

# Read data ...

# Define GP model.
gp_model_code = """
data {
    int D;               // number of features (dimensions of X)
    int N;               // number of observations
    vector[N] y;         // response
    matrix[N, D] X;      // predictors
    
    // hyperparameters for GP covariance function range and scale.
    real m_rho;
    real<lower=0> s_rho;
    real m_alpha;
    real<lower=0> s_alpha;
    real m_sigma;
    real<lower=0> s_sigma;
}

transformed data {
    // GP mean function.
    vector[N] mu = rep_vector(0, N);
}

parameters {
    real<lower=0> rho;   // range parameter in GP covariance fn
    real<lower=0> alpha; // covariance scale parameter in GP covariance fn
    real<lower=0> sigma;   // model sd
}

model {
    matrix[N, N] K;   // GP covariance matrix
    matrix[N, N] LK;  // cholesky of GP covariance matrix

    rho ~ lognormal(m_rho, s_rho);  // GP covariance function range parameter
    alpha ~ lognormal(m_alpha, s_alpha);  // GP covariance function scale parameter
    sigma ~ lognormal(m_sigma, s_sigma);  // model sd.
   
    // Using exponential quadratic covariance function
    // K(d) = alpha^2 * exp(-0.5 * (d/rho)^2)
    K = cov_exp_quad(to_array_1d(X), alpha, rho); 
    
    // Add small values along diagonal elements for numerical stability.
    for (n in 1:N) {
        K[n, n] = K[n, n] + sigma^2;
    }
        
    // Cholesky of K (lower triangle).
    LK = cholesky_decompose(K);

    // GP likelihood.
    y ~ multi_normal_cholesky(mu, LK);
}
"""


# Compile model. This takes about a minute.
sm = pystan.StanModel(model_code=gp_model_code)

# Data dictionary.
data = dict(y=y, X=X, N=y.shape[0], D=1,
            m_rho=0, s_rho=1.0, m_alpha=0, s_alpha=0.1, m_sigma=0, s_sigma=1)

# Fit via ADVI.
vb_fit = sm.vb(data=data, iter=2000, seed=2, grad_samples=1, elbo_samples=1)
vb_samples = pystan_vb_extract(vb_fit)

# Fit via HMC
hmc_fit = sm.sampling(data=data, iter=2000, chains=1, warmup=1000, thin=1,
                      seed=1, algorithm='HMC',
                      control=dict(stepsize=0.01, int_time=1))

# Fit via NUTS
nuts_fit = sm.sampling(data=data, iter=2000, chains=1, warmup=1000, thin=1,
                       seed=1)
