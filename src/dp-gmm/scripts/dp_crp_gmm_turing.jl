import Dates
println("Last updated: ", Dates.now(), " (PT)")

# Load environment
import Pkg; Pkg.activate("../../../")

# Import Libraries
using Turing
using Turing: Variational
import Turing.RandomMeasures.DirichletProcess
import Turing.RandomMeasures.ChineseRestaurantProcess
using Distributions
using JSON3
using PyPlot
using StatsFuns
import Random
import StatsBase.countmap
include(joinpath(@__DIR__, "../util/BnpUtil.jl"));

# Directory where all simulation data are stored.
data_dir = joinpath(@__DIR__, "../../data/sim-data")
# path_to_data = joinpath(data_dir, "gmm-data-n50.json")
path_to_data = joinpath(data_dir, "gmm-data-n200.json")

# Load data in JSON format.
data = let
    x = open(f -> read(f, String), path_to_data)
    JSON3.read(x, Dict{Symbol, Vector{Any}})
end

# Convert data to vector of floats
y = Float64.(data[:y]);

# Visualize data
plt.hist(y, bins=50, density=true)
plt.xlabel("y")
plt.ylabel("density")
plt.title("Histogram of data");

function counts_plot(x; density=false, color="C0", lw=4)
    cm_x = countmap(x)
    number = collect(keys(cm_x))
    count = let
        c = collect(values(cm_x))
        density ? c / sum(c) : c
    end
    plt.vlines(number, 0, count, color=color, lw=lw)
end

# DP GMM model under CRP construction
@model dp_gmm_crp(y) = begin
    nobs = length(y)
    
    alpha ~ Gamma(1, 0.1)  # mean = a*b
    rpm = DirichletProcess(alpha)
    
    # Base measure.
    H = arraydist([Normal(0, 3), InverseGamma(2, 0.05)])
    
    # Latent assignment.
    z = tzeros(Int, nobs)
    
    # Locations and scales of infinitely many clusters.
    mu_sigma = TArray(Vector{Float64}, 0)
    
    for i in 1:nobs
        # Number of clusters.
        K = maximum(z)
        n = Vector{Int}([sum(z .== k) for k in 1:K])
        
        # Sample cluster label.
        z[i] ~ ChineseRestaurantProcess(rpm,  n)
        
        # Create a new cluster.
        if z[i] > K
            push!(mu_sigma, [0.0, 0.1])
            mu_sigma[z[i]] ~ H
        end
        
        # Sampling distribution.
        mu, sigma = mu_sigma[z[i]]
        y[i] ~ Normal(mu, sigma)
    end
end
;

# Set random seed for reproducibility
Random.seed!(0);

# Compile time approx. 32s.
# Run time approx. 70s

@time chain = begin
    burn = 2000  # NOTE: The burn in is also returned. Discard manually.
    n_samples = 1000
    iterations = burn + n_samples

    sample(dp_gmm_crp(y), SMC(), iterations)
    # sample(dp_gmm_crp(y), IS(), iterations)
    # sample(dp_gmm_crp(y), Gibbs(PG(5, :z), PG(5, :mu_sigma)), iterations)
    # sample(dp_gmm_crp(y), PG(10), iterations)
end;

chain.name_map
chain.value
# mean(chain[:mu_sigma].value.data[:, [1,3,5,7,9,11], 1], dims=1)

z = chain.value.data[end-burn+1:end, (end-length(y)+1):end, 1]
nclus = [length(unique(z[i, :])) for i in 1:size(z, 1)];
plt.figure(figsize=(12,4))
plt.subplot(1, 3, 1)
plt.plot(nclus);
plt.subplot(1, 3, 2)
counts_plot(nclus);
plt.subplot(1, 3, 3)
counts_plot(z[end, :]);

# chain.value.data[:, 4:10, 1]

# exp_num_clus(a, n) = a * log1p(n / a)
# counts_plot(rand(Poisson(exp_num_clus(1.0, 200)), 100000), density=true);


