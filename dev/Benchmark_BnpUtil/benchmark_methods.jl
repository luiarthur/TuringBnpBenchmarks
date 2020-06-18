import Pkg; Pkg.activate("../../")

using StatsFuns
using BenchmarkTools
import Random
include(joinpath(@__DIR__, "BnpUtil_old.jl"))

function log_sum_exp(logx::T) where T <: AbstractArray
  mx = maximum(logx)
  return mx + log(sum(exp.(logx .- mx)))
end

Random.seed!(0);
x = randn(10)
@assert StatsFuns.logsumexp(x) == BnpUtil.logsumexpdd(x)[1]

# Logsumexp benchmarks
@benchmark StatsFuns.logsumexp(x)
@benchmark BnpUtil.logsumexp(x)[1]
@benchmark log_sum_exp(x)

# Gmm benchmarks
Random.seed!(0);
N, K = 101, 10
X = randn(N, K)
m = randn(K)
s = rand(K)
w = let
    _w = rand(K)
    _w / sum(_w)
end

lpdf_gmm_sf(x, m, s, w) = sum(logsumexp(normlogpdf.(m, s, x) .+ log.(w), dims=2))
@assert sum(BnpUtil.lpdf_gmm(X, m', s', w', dims=2)) == lpdf_gmm_sf(X, m, s, w)
@btime lpdf_gmm_sf(X, m', s', w')
@btime sum(BnpUtil.lpdf_gmm(X, m', s', w', dims=2))

# NOTE: 
# logsumexp and lpdf_gmm seems to be comparable. In fact, for forward
# computaton, StatsFuns seems to be a little faster for logsumexp.
