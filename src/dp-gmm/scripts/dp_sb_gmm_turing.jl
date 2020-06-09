import Dates
println("Last updated: ", Dates.now(), " (PT)")

# Load environment
import Pkg; Pkg.activate("../../../")

# Import Libraries
using Turing
using Distributions
using JSON3
using PyPlot
using StatsFuns
import Random
import StatsBase: countmap
include(joinpath(@__DIR__, "../util/BnpUtil.jl"));

# DP GMM model under stick-breaking construction
@time @model dp_gmm_sb(y, K) = begin
    nobs = length(y)

    mu ~ filldist(Normal(0, 3), K)
    sig ~ filldist(Gamma(1, 1/10), K)  # mean = 0.1
    
    alpha ~ Gamma(1, 1/10)  # mean = 0.1
    v ~ filldist(Beta(1, alpha), K - 1)
    eta = BnpUtil.stickbreak(v)

    # NOTE: Slow. And the MCMC gets stuck?
    # y .~ MixtureModel(Normal.(mu, sig), eta)

    # NOTE: Fast, and seems to mix well.
    log_target = logsumexp(normlogpdf.(mu', sig', y) .+ log.(eta)', dims=2)
    Turing.acclogp!(_varinfo, sum(log_target))
end

# Directory where all simulation data are stored.
data_dir = joinpath(@__DIR__, "../../data/sim-data")
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

# Fit DP-SB-GMM

# Set random seed for reproducibility
Random.seed!(0);

# NUTS
# Compile time approx. 11s
# Run time approx. 244s
# Slower, but works a little better.
@time chain = begin
    n_components = 10
    n_samples = 500
    nadapt = 500
    iterations = n_samples + nadapt
    burn = 0  # For compatibility with HMC below.
    target_accept_ratio = 0.8
    
    sample(dp_gmm_sb(y, n_components),
           NUTS(nadapt, target_accept_ratio),
           iterations);
end

# HMC
# Compile time approx. 32s.
# Run time approx. 70s.
#
# @time chain = begin
#     burn = 500  # NOTE: The burn in is also returned. Can't be discarded.
#     n_samples = 500
#     iterations = burn + n_samples
#     n_components = 10
#     stepsize = 0.01
#     nleapfrog = floor(Int, 1 / stepsize)
#  
#     sample(dp_gmm_sb(y, n_components), 
#            HMC(stepsize, nleapfrog),
#            iterations)
# end

function extract(chain, sym; burn=0)
    tail  = chain[sym].value.data[(burn + 1):end, :, :]
    return dropdims(tail, dims=3)
end

vpost = extract(chain, :v, burn=burn);
mupost = extract(chain, :mu, burn=burn);
sigpost = extract(chain, :sig, burn=burn);
etapost = hcat([BnpUtil.stickbreak(vpost[row, :]) for row in 1:size(vpost, 1)]...)';

function plot_param_post(param, param_name, param_full_name; figsize=(10, 4), burn=0)
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.boxplot(param, whis=[2.5, 97.5], showmeans=true, showfliers=false)
    plt.xlabel("mixture components")
    plt.ylabel(param_full_name)
    plt.title("95% Credible Intervals for $(param_full_name)")

    plt.subplot(1, 2, 2)
    plt.plot(param)
    plt.xlabel("iterations")
    plt.ylabel(param_full_name)
    plt.title("Trace plot of $(param_full_name)");
end

plot_param_post(etapost, :eta, "mixture weights (η)", burn=burn)

plot_param_post(mupost, :mu, "mixture means (μ)", burn=burn)

plot_param_post(sigpost, :sigma, "mixture scales (σ)", burn=burn)

# TODO: How to get loglikelihood or log posterior?

# Plot posterior distribution of number of clusters

# Set a threshold for clusters to be considered as significant.
thresh = 0.01

plt.figure(figsize=(10, 4))

# Trace plot
plt.subplot(1, 2, 1)
plt.plot(sum(etapost .> thresh, dims=2))
plt.xlabel("iteration")
plt.ylabel("Number of components > $thresh")
plt.title("Trace plot of number of active components");

# Bar plot
plt.subplot(1, 2, 2)
ncomponents_post = vec(sum(etapost .> thresh, dims=2))
num_samples = length(ncomponents_post)
countmap_ncomponents = countmap(ncomponents_post)
x_ncomp = Int.(keys(countmap_ncomponents))
y_prop = Int.(values(countmap_ncomponents)) / num_samples
plt.bar(x_ncomp, y_prop)
plt.xlabel("Number of active components")
plt.ylabel("Posterior probability")
plt.title("Distribution of number of active components");

plt.hist(vec(chain[:alpha].value), density=true, bins=30)
plt.xlabel("α")
plt.ylabel("density")
plt.title("Histogram of mass parameter α");
