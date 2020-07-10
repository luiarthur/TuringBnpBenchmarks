import Pkg; Pkg.activate("../../")
using Turing

# Simulate data.
N = 1000
mu_true = [5.0 10.0 2.0]
sigma_true = 0.6
y = randn(N, length(mu_true)) * sigma_true .+ mu_true  # size(y) = (1000, 3)

# Model definition.
@model function demo_model(y)
    K = size(y, 2)
    mu ~ filldist(Normal(0, 10), K)
    sigma ~ LogNormal(0, 2)
    for k in 1:K
        y[:, k] .~ Normal(mu[k], sigma)
    end
end

# Sample from posterior via NUTS.
nadapt, target_accept_ratio = (1000, 0.8)
iterations = 1000 + nadapt
chain = sample(demo_model(y), NUTS(nadapt, target_accept_ratio), iterations)

# Get sigma
chain[:sigma]  # works for univariate parameters
# 2-dimensional AxisArray{Float64,2,...} with axes:
#     :iter, 1:1:1000
#     :chain, 1:1
# And data, a 1000×1 Array{Float64,2}:
#  0.5891087703064164
#  0.5982078191819813
#  0.6014525731322462
#  0.6049135394153814
#  0.6135096911950264
#  ⋮
#  0.6045248767595608
#  0.6043856761562517
#  0.6197436168542185
#  0.6036546321663407

# FIXME: Get mu
# chain[:mu]  # broken

# Workarouund for getting array parameters
# function extract_array_param(chain, sym)
#     rgx = Regex("$(sym)\\[\\d+(,\\d+)*\\]")
#     syms = filter(x -> x != nothing, match.(rgx, String.(chain.name_map[1])))
#     syms = map(x -> Symbol(x.match), syms)
#     return chain[syms]
# end
# 
# extract_array_param(chain, :mu)

# This has been fixed.
group(chain, :mu)

# FIXME: Get log-likelihood.
loglike = logprob"y=y | model=demo_model, chain=chain"  # broken
