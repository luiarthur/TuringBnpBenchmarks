# NOTE: This script is used for generating data for the PPL comparisons.

import Pkg; Pkg.activate("../../../")

using Distributions
using JSON3
import Random

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

function main(; seed=1, num_components=4, 
              datasizes=[50, 100, 200, 400, 800, 1600],
              data_dir=joinpath(@__DIR__, "../data"))
  Random.seed!(seed)

  # simulate mu
  mu = let
      _mu = collect(1:num_components)
      centered_mu = _mu .- (num_components + 1) / 2 
      jittered_mu = centered_mu .+ randn(num_components) * .1
      jittered_mu
  end

  # simulate sigma
  sig = rand(Uniform(.1, .2), num_components)

  # simulate weights
  w = let
      w_tmp = rand(Uniform(.8/num_components, 1.2/num_components),
                   num_components)
      w_tmp ./ sum(w_tmp)
  end

  # Create data sets and write to disk
  for n in datasizes
      # Generate GMM data.
      gmm_data = generate_gmm_data(n=n, mu=mu, sig=sig, w=w, seed=seed)
    
      # Write GMM data to json string.
      gmm_data_json = JSON3.write(gmm_data);
    
      # Write json string to disk.
      open("$(data_dir)/gmm-data-n$(n).json", "w") do io
        write(io, gmm_data_json)
      end
  end
end

# Generate data.
main()
