module Kernels

using Distances
import LinearAlgebra

export SqExpKernel, metric

abstract type DistKernel end

struct SqExpKernel <: DistKernel
  alpha
  rho
  eps
end

SqExpKernel(alpha::Real, rho::Real) = SqExpKernel(alpha, rho, 1e-8)
metric(k::SqExpKernel) = SqEuclidean()

function (k::SqExpKernel)(Dsq::AbstractMatrix; jitter=true)
  out = k.alpha^2 * exp.(-Dsq / (2 * k.rho^2))
  if jitter
    out += LinearAlgebra.I * k.eps
  end
  return out
end

end
