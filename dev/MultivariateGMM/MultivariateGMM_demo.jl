using Distributions
using Turing
using PyPlot
import LinearAlgebra

eye(n::Int) = Matrix{Float64}(LinearAlgebra.I, n, n)

# Model specification.
@model MultivariateGMM(Y, K, ::Type{T}=Vector{Matrix{Float64}}) where T = begin
  N, D = size(Y)

  # Priors.
  w ~ Dirichlet(K, 1/K)
  mu ~ filldist(Normal(0, 1), D, K)
  Sigma = T(undef, K)
  for k in 1:K
    Sigma[k] ~ InverseWishart(D+2, eye(D))
  end

  # Likelihood.
  for i in 1:N
    Y[i, :] ~ MixtureModel([MvNormal(mu[:, k], Sigma[k]) for k in 1:K], w)
  end
end

# Genreate data.
Y = [randn(50, 2) .+ 3; randn(30, 2) .- 3]

# Fit model.
K = 5
m = MultivariateGMM(Y, K)
nburn, nsamps = 200, 200
@time chain = sample(m, NUTS(nburn, 0.8, max_depth=5), nburn + nsamps)

# Collect posterior samples.
mu = reshape(group(chain, :mu).value.data[:, :, 1], nsamps, 5, 2);
Sigma = reshape(group(chain, :Sigma).value.data, nsamps, 2, 2, 5);
w = group(chain, :w).value.data[:, :, 1];

# Posterior predictive.
postpred = let
  [let
     mm = MixtureModel([MvNormal(mu[i, k, :], Sigma[i, :, :, k]) for k in 1:K],
                       w[i, :])
     rand(mm)
   end for i in 1:nsamps]
end
postpreds = hcat(postpred...)

# Plot posterior predictive.
plt.scatter(postpreds[1, :], postpreds[2, :], label="Post. predictive")
plt.scatter(Y[:, 1], Y[:, 2], c="red", label="Data")
plt.legend();

# Plot w.
plt.boxplot(w);

# Plot log unnormalized joint density.
plt.plot(get(chain, :log_density)[1].data[:, 1])
plt.title("Trace of log prob");
