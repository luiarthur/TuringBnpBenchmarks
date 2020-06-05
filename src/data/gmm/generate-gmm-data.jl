"""
Function to generate data for univariate Gaussian mixture model (GMM).

Args
====

- `n`: number of observations
- `mu`: Mixture component means
- `sig`: Standard deviations of each component
- `w`: Mixture component weights
- `seed`: Random number generator seed
"""
function generate_gmm_data(; n::Integer,
                           mu::Vector{Float64}, sig::Vector{Float64},
                           w::Vector{Float64},
                           seed=nothing) where T <: AbstractFloat
    if seed != nothing
        Random.seed!(seed)
    end

    # Number of mixture components
    num_components = length(mu)

    # Check that w is a proper probability vector
    @assert all(w .> 0) && sum(w) == 1

    # Generate random class labels
    rand_idx = wsample(1:num_components, w, n)

    # Generate random observations
    y = mu[rand_idx] + randn(n) .* sig[rand_idx]

    return Dict(:rand_idx => rand_idx,
                :mu => mu, :sig => sig, :w => w, :y => y)
end
