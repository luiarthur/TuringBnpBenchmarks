data {
  int<lower=0> N;
  int<lower=0> M;
  int<lower=0> P;
  vector[N] y;
  matrix[N, P] X;
  matrix[M, P] W;
  real m_K;
  real<lower=0> s_K;
}

transformed data {
  matrix[N, M] D;  // Distance matrix.

  // Create distance matrix.
  for (n in 1:N) for (m in 1:M) {
    D[n, m] = distance(X[n, :], Y[m, :]);
  }
}

parameters {
  vector[N] z;  // White noise process.
  real<lower=0> sigma_K;  // Range parameter in covariance.
}

transformed parameters {
  // Latent function.
  vector[N] f = K * z;
}

model {
  matrix[N, M] K;  // Kernel matrix.

  // Priors.
  z ~ normal(0, 1);
  sigma_K ~ lognormal(0, 1);

  // Populate kernel matrix.
  for (n in 1:N) for (m in 1:M) {
    K[n, m] = exp(normal_lpdf(D[n, m] | 0, sigma_K));
  }

  // Model.
  y ~ bernoulli_logit(f);
}
