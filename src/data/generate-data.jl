import Pkg; Pkg.activate("../../")

using Distributions
import LinearAlgebra
import Random
using JSON3

include(joinpath(@__DIR__, "gmm/generate-gmm-data.jl"))

eye(n::Integer) = Matrix(LinearAlgebra.I, n, n)

# MAIN
seed = 1
Random.seed!(seed)

num_components = 4
mu = let
    _mu = collect(1:num_components)
    centered_mu = _mu .- (num_components + 1) / 2 
    jittered_mu = centered_mu .+ randn(num_components) * .1
    jittered_mu
end
sig = rand(Uniform(.1, .2), num_components)
w = let
    w_tmp = rand(Uniform(.8/num_components, 1.2/num_components),
                 num_components)
    w_tmp ./ sum(w_tmp)
end

# Directory to data
data_dir = joinpath(@__DIR__, "sim-data")
mkpath(data_dir)

# Data sizes to use for simulation studies
ns = [50, 100, 200, 400, 800, 1600]

# Create data sets and write to disk
for n in ns
    # Generate GMM data.
    gmm_data = generate_gmm_data(n=n, mu=mu, sig=sig, w=w, seed=seed)
  
    # Write GMM data to json string.
    gmm_data_json = JSON3.write(gmm_data);
  
    # Write json string to disk.
    open("$(data_dir)/gmm-data-n$(n).json", "w") do io
      write(io, gmm_data_json)
    end
end

# NOTE: To read data, do the following:
# json = open(f -> read(f, String), "$(data_dir)/gmm-data-n50.json")
# JSON3.read(json, Dict{Symbol, Vector{Any}})
