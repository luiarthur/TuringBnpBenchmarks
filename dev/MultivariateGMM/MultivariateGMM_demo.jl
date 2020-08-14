using Distributions
using Turing
using PyPlot
import LinearAlgebra
import Random
using StatsFuns
using Flux

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
Random.seed!(0)
Y = [rand(MvNormal([-3, -3], [[1., 0.8] [0.8, 1.]]), 50)';
     rand(MvNormal([3, 3], [[1., -0.7] [-0.7, 1.]]), 30)']

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

# Predict cluster membership at new location.
function predict(x::AbstractVector, mu, Sigma, w)
  K = length(w)
  ll = [logpdf(MvNormal(mu[k, :], Sigma[:, :, k]), x) for k in 1:K]
  logp = ll + log.(w)
  return exp.(logp .- logsumexp(logp))
end

function postpredict(x::AbstractVector, mus, Sigmas, ws)
  nsamps, K = size(ws)
  return [let 
     p = predict(x, mus[i, :, :], Sigmas[i, :, :, :], ws[i, :, :])
     wsample(1:K, vec(p))
   end for i in 1:nsamps]
end

# Plot squares.
_xgrid = [[i, j] for i in range(-6,6,length=50), j in range(-6,6,length=50)]
xgrid = vec(_xgrid)
@time pp = hcat([postpredict(x, mu, Sigma, w) for x in xgrid]...);
pp_onehot = [Flux.onehot(pp[i, j], 1:5) for i in 1:size(pp,1), j in 1:size(pp,2)]
meanprobs_onehot = [mean(pp_onehot[:, j]) for j in 1:size(pp_onehot,2)]
stdprobs_onehot = [std(pp_onehot[:, j]) for j in 1:size(pp_onehot,2)]

xg = hcat(xgrid...)
colors = ["blue", "yellow", "orange", "green", "red"]
label = colors[argmax.(meanprobs_onehot)]
alpha = maximum.(meanprobs_onehot)

# Cluster predictions.
gs = [[g[i] for g in _xgrid] for i in 1:2]
Colors = ["Blues", "YlOrBr", "Oranges", "Greens", "Reds"]
for k in 1:K
  idx = findall(label .!== colors[k])
  _alpha = copy(reshape(alpha, size(gs[1])))
  _alpha[idx] .= NaN
  plt.contourf(gs[1], gs[2], _alpha, 101, cmap=Colors[k], vmin=0, vmax=1)
end
plt.title("Cluster Predictions")
plt.scatter(Y[:, 1], Y[:, 2], color="grey", edgecolor="black")

# Cluster uncertainty.
uq = [sqrt(sum(s.^2)) for s in stdprobs_onehot]
uq = reshape(uq, size(gs[1]))
plt.contourf(gs[1], gs[2], uq, 101, cmap="Oranges", vmin=0)
plt.colorbar()
plt.title("Cluster Uncertainty")
plt.scatter(Y[:, 1], Y[:, 2], color="grey", edgecolor="black")
