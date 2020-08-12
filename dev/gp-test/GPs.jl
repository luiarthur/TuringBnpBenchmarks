# https://gregorygundersen.com/blog/2019/09/12/practical-gp-regression/
module GPs

using Distributions
using Distances

include(joinpath(@__DIR__, "Kernels.jl"))
using .Kernels

import LinearAlgebra
import LinearAlgebra: cholesky, Symmetric

export metric, SqExpKernel, GP, posterior, Kernels

struct GP
  mu
  kernel
  y
  X
  sigma
  C
  d
  a
end

function GP(y, X, kernel, mu=0, sigma=0)
  Ddata = pairwise(metric(kernel), X, dims=1)
  Kdata = kernel(Ddata)
  C = cholesky(Symmetric(Kdata))
  d = y .- mu
  a = C \ d
  return GP(mu, kernel, y, X, sigma, C, d, a)
end

function posterior(gp, Xnew)
  Dnew = pairwise(metric(gp.kernel), Xnew, dims=1)
  D_new_data = pairwise(metric(gp.kernel), Xnew, gp.X, dims=1)

  Knew = gp.kernel(Dnew)
  K_new_data = gp.kernel(D_new_data, jitter=false)

  mu = gp.mu .+ K_new_data * gp.a
  S = Knew - K_new_data * (gp.C \ K_new_data')

  return MvNormal(mu, Symmetric(S))
end

end
