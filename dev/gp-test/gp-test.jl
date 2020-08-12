import Pkg; Pkg.activate("../../")

include("GPs.jl")

using Distributions
using PyPlot
using BenchmarkTools
import Random

Random.seed!(4)
n = 10
x = randn(n)
f(x) =  sin(3 * x) * sin(x) * (-1)^(x > 0)
sigma = 0
y = f.(x) + randn(n) * sigma;

xgrid = range(minimum(x) - 0.3, maximum(x) + 0.3, length=100)
rhos = rand(Uniform(0.4, 0.6), 50)

p = [let
    gp = GPs.GP(y, x[:,:], GPs.SqExpKernel(1, rho), sigma=0);
    post = GPs.posterior(gp, xgrid[:,:])
    rand(post)
end for rho in rhos]
p = hcat(p...)

plt.scatter(x, y)
p_lower, p_upper = GPs.ci(p)
plt.plot(xgrid, mean(p, dims=2))
plt.fill_between(xgrid, p_lower, p_upper, alpha=0.2, color="blue")

for i in sample(1:size(p, 2), 10)
    plt.plot(xgrid, p[:, i], alpha=0.2, color="black")
end

plt.savefig("out/post.pdf")
plt.close()
