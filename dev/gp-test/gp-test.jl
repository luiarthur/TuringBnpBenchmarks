import Pkg; Pkg.activate("../../")

include("GPs.jl")
using .GPs

using Distributions
using PyPlot
using BenchmarkTools
import Random

Random.seed!(4)
x = randn(100)
f(x) =  sin(3 * x) * sin(x) * (-1)^(x > 0)
y = f.(x);

xgrid = range(-3, 3, length=100)
@time gp = GP(y, x[:,:], SqExpKernel(1, 0.5));
@time post = posterior(gp, xgrid[:,:])
p = rand(post, 50)
plt.scatter(x, y)
plt.plot(xgrid, mean(p, dims=2))
plt.savefig("bla.pdf")
plt.close()
