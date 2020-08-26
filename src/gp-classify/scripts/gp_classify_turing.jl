print("Last updated: $(read(`date`, String))")

# Load environment
import Pkg; Pkg.activate("../../../")

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
function sekernel(alpha, rho, jitter=0)
  kernel = alpha^2 * transform(SEKernel(), invsqrt2/rho)
  jitter <= 0 || (kernel += jitter * EyeKernel())
  return kernel
end

function compute_f(kernel, X, eta, beta=0, jitter=0)
  K = kernelmatrix(kernel, X, obsdim=1)
  jitter <= 0 || (K += LinearAlgebra.I * jitter)
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
    y ~ Product(Bernoulli.(logistic.(f)))
end;

# For getting quantiles along array dimensions
quantiles(X, q; dims) = mapslices(x -> quantile(x, q), X, dims=dims)

# To extract parameters from ADVI model.
function make_extractor(m, q; nsamples=300)
    qsamples = rand(q, nsamples)
    _, sym2range = Variational.bijector(m; sym_to_ranges=Val(true))
    return sym -> qsamples[collect(sym2range[sym][1]), :]
end;

function plot_data(X, y; edgecolors=nothing)
    colors = ["blue", "red"]
    plt.scatter(X[:, 1], X[:, 2], color=colors[y .+ 1], edgecolors=edgecolors)
end;

# Function for plotting parameter posterior.
function plot_post(samples, name; bins=nothing, suffix="")
    param = samples[name]
    plt.hist(param, density=true, bins=bins)
    plt.xlabel(name)
    plt.ylabel("density")
    if suffix !== ""
        suffix = "($(suffix))"
    end
    plt.title("Histogram of $(name) $(suffix)");
end;

function makegrid(xmin, xmax, ymin, ymax, ngrid; return_all=false)
  ngrid = floor(Int, sqrt(ngrid))
  grid = [[i, j]
          for i in range(xmin, xmax, length=ngrid),
          j in range(ymin, ymax, length=ngrid)]
  out =  Matrix(hcat(vec(grid)...)')
    
  if return_all
    return out, grid
  else
    return out
  end
end;

function gp_predict(postsamps, X, y, nnew; eps=0, prob=true, jitter=1e-6)
    # Make grid of prediction locations.
    xmin = minimum(X, dims=1)
    xmax = maximum(X, dims=1)
    Xnew, grid = makegrid(xmin[1]-eps, xmax[1]+eps,
                          xmin[2]-eps, xmax[2]+eps,
                          nnew, return_all=true)
    
    # Posterior samples of model parameters.
    alpha = postsamps[:alpha]
    rho = postsamps[:rho]
    beta = postsamps[:beta]
    nsamps = length(alpha)
    eta = postsamps[:eta]
        
    # Predict classification probabilities.
    ps = [let
            kernel = sekernel(alpha[i], rho[i], jitter)
            gp = GP(beta[i], kernel)
            f = compute_f(kernel, X, eta[:, i], beta[i])
            pfx = posterior(gp(X'), f)
            logit_p = rand(pfx(Xnew'))
            prob ? logistic.(logit_p) : logit_p
         end for i in ProgressBar(1:nsamps)]

    return hcat(ps...), Xnew, grid
end;

function plot_kernel_params(post_samples, kernel_params; bins=nothing, figsize=(8,3))
    plt.figure(figsize=figsize)
    nparams = length(kernel_params)
    for i in 1:nparams
        param = kernel_params[i]
        plt.subplot(1, nparams, i)
        plot_post(post_samples, param, bins=bins);
    end
    plt.tight_layout()
end;

function plot_uq(postsamps, X, y, algo;
                 nnew=100, eps=0.2, figsize=(10, 8),
                 kernel_params=nothing, jitter=1e-6,
                 color_res=100, color_res_sd=100, return_stuff=false,
                 data_edgecolors=nothing, vmin_sd=0, vmax_sd=nothing, prob=true)
    nsamps = postsamps[:alpha]
    ps, Xnew, grid = gp_predict(postsamps, X, y, nnew, eps=eps, prob=prob, jitter=jitter)
    
    plt.figure(figsize=figsize)
    gs = [[g[i] for g in grid] for i in 1:2]
    
    # Plot mean function.
    plt.subplot(2, 2, 1)
    if prob
        plt.contourf(gs[1], gs[2], reshape(mean(ps, dims=2), size(gs[1])...),
                     color_res, cmap=plt.get_cmap("bwr"), vmin=0, vmax=1)
    else
        plt.contourf(gs[1], gs[2], reshape(mean(ps, dims=2), size(gs[1])...),
                     color_res, cmap=plt.get_cmap("bwr"))
    end
    plt.colorbar();
    plot_data(X, y, edgecolors=data_edgecolors);
    plt.xticks([]); plt.yticks([]);
    plt.title("Posterior Mean Function ($(algo))")
    
    # Plot function sd.
    plt.subplot(2, 2, 2)
    plt.contourf(gs[1], gs[2], reshape(std(ps, dims=2), size(gs[1])...),
                 color_res_sd, cmap=plt.get_cmap("Oranges"),
                 vmin=vmin_sd, vmax=vmax_sd)
    plt.colorbar();
    plot_data(X, y)
    plt.xticks([]);
    plt.yticks([]);    
    plt.title("Posterior Standard Deviation Function ($algo)")
    
    if kernel_params != nothing
        plot_kernel_params(postsamps, kernel_params, bins=30)
    end
   
    if return_stuff
        return ps, Xnew, grid
    end
end;

# Read data.

# Path to data.
data_path = joinpath(@__DIR__, "../data/gp-classify-data-N50.json")

# Load data in JSON format.
data = let
    x = open(f -> read(f, String), data_path)
    JSON3.read(x)
end

# Store data.
X = [data["x1"] data["x2"]]
y = Int64.(data["y"])

# Plot data.
plot_data(X, y);

# Create model.
m =  GPClassify(Float64.(y), X);
kernel_params = [:alpha, :rho, :beta];

# Fit via ADVI.
Random.seed!(7)

# NOTE: ADVI(num_elbo_samples, max_iters)

# Compile
@time q = vi(m, ADVI(1, 1))

# RUN
@time q = vi(m, ADVI(1, 1000))
 
# Get posterior samples
extract_gp = make_extractor(m, q, nsamples=500)
advi_samples = Dict{Symbol, Any}(sym => vec(extract_gp(sym)) for sym in kernel_params)
advi_samples[:eta] = extract_gp(:eta);

plot_uq(advi_samples, X, y, "ADVI", eps=0.5,
        nnew=500, data_edgecolors="gold",
        kernel_params=kernel_params,
        color_res=range(0, 1, length=101),
        color_res_sd=range(0, .5, length=101));

# Fit via HMC.
Random.seed!(0)
burn = 500
nsamples = 500

# Compile
@time _ = sample(m, HMC(0.01, 1), 1)

# Run
@time hmc_chain = sample(m, HMC(0.05, 20), burn + nsamples);

# Get posterior samples
hmc_samples = Dict{Symbol, Any}([
    sym => vec(group(hmc_chain, sym).value.data)[end-nsamples+1:end]
for sym in kernel_params])
hmc_samples[:eta] = Matrix(group(hmc_chain, :eta).value.data[end-nsamples+1:end, :, 1]');

# Plot HMC
ps, Xnew, grid = plot_uq(hmc_samples, X, y, "HMC", eps=0.5,
                         nnew=500, data_edgecolors="gold",
                         kernel_params=kernel_params,
                         color_res=range(0, 1, length=101),
                         return_stuff=true);

plt.plot(get(hmc_chain, :lp)[1].data[burn+1:end]);
plt.xlabel("HMC iterations");
plt.ylabel("log unnormalized joint density");

# Fit via NUTS.
Random.seed!(0)
burn = 500
nsamples = 500

# Compile
@time _ = sample(m, NUTS(1, 0.8, max_depth=2), 2)

# Run
@time nuts_chain = sample(m, NUTS(burn, 0.8, max_depth=10), burn + nsamples);

# Get posterior samples
nuts_samples = Dict{Symbol, Any}([
    sym => vec(group(nuts_chain, sym).value.data)
for sym in kernel_params])
nuts_samples[:eta] = Matrix(group(nuts_chain, :eta).value.data[:, :, 1]');

# Plot NUTS
plot_uq(nuts_samples, X, y, "NUTS", eps=0.5,
        nnew=500, data_edgecolors="gold",
        kernel_params=kernel_params,
        color_res=range(0, 1, length=101),
        color_res_sd=range(0, .5, length=101),
        return_stuff=true);

plt.plot(get(nuts_chain, :lp)[1].data);
plt.xlabel("NUTS iterations");
plt.ylabel("log unnormalized joint density");


