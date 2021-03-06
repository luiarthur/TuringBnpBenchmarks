# Import Libraries
using Turing
using Turing: Variational
using Distributions
using AbstractGPs, KernelFunctions
using PyPlot
using StatsFuns
using JSON3
using Flux
using ProgressBars
import Random
import LinearAlgebra

# Define a kernel.
sekernel(alpha, rho) = alpha^2 * transform(SEKernel(), invsqrt2/rho)

function compute_f(kernel, X, eta, beta=0, jitter=0)
  K = kernelmatrix(kernel, X, obsdim=1) + LinearAlgebra.I * jitter
  return LinearAlgebra.cholesky(K).L * eta .+ beta
end

@model function GPClassify(y, X, jitter=1e-6)
    # Priors.
    alpha ~ LogNormal(0, 1)
    rho ~ LogNormal(0, 1)
    beta ~ Normal(0, 1)  # intercept.
    eta ~ filldist(Normal(0, 1), length(y))

    # Latent GP
    kernel = sekernel(alpha, rho)
    f = compute_f(sekernel(alpha, rho), X, eta, beta, jitter)
    
    # Sampling Distribution.
    y ~ arraydist(Bernoulli.(logistic.(f)))
end;


# To extract parameters from trained variational distribution
# (surrogate posterior).
function make_surrogate_sampler(m, q; nsamples=300)
    qsamples = rand(q, nsamples)
    _, sym2range = Variational.bijector(m; sym_to_ranges=Val(true))
    return sym -> qsamples[collect(sym2range[sym][1]), :]
end;

### Read data ###
# See data in Notebook.
data_path = joinpath(@__DIR__, "../data/gp-classify-data-N50.json")

# Load data in JSON format.
data = let
    x = open(f -> read(f, String), data_path)
    JSON3.read(x)
end

### Name the data (X, y) ###
X = [data["x1"] data["x2"]]
y = Int64.(data["y"])

### Create model.###
m =  GPClassify(Float64.(y), X);
kernel_params = [:alpha, :rho, :beta];

### Fit via ADVI ###
Random.seed!(7)
@time q = vi(m, ADVI(1, 1000))  # num_elbo_samples, num_iterations.
 
# Get posterior samples
surrogate_sampler = make_surrogate_sampler(m, q, nsamples=500)
advi_samples = Dict{Symbol, Any}(sym => vec(surrogate_sampler(sym))
                                 for sym in kernel_params)
advi_samples[:eta] = extract_gp(:eta);

### Fit via HMC ###
Random.seed!(0)
burn = 500
nsamples = 500
@time hmc_chain = sample(m, HMC(0.05, 20), burn + nsamples);

# Get posterior samples
hmc_samples = Dict{Symbol, Any}([
    sym => vec(group(hmc_chain, sym).value.data)[end-nsamples+1:end]
for sym in kernel_params])
hmc_samples[:eta] = Matrix(group(hmc_chain, :eta).
                           value.data[end-nsamples+1:end, :, 1]')

### Fit via NUTS ###
Random.seed!(0)
burn = 500
nsamples = 500
@time nuts_chain = sample(m, NUTS(burn, 0.8, max_depth=10), burn + nsamples)

# Get posterior samples
nuts_samples = Dict{Symbol, Any}([
    sym => vec(group(nuts_chain, sym).value.data)
for sym in kernel_params])
nuts_samples[:eta] = Matrix(group(nuts_chain, :eta).value.data[:, :, 1]')
