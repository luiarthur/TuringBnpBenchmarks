module BnpUtil


"""
This is a stable implementation of `log(sum(exp(logx)))`. 

Arguments
=========
- `logx`: An abstract array of arbitrary dimensions
- `dims`: The dimension along which to perform logsumexp. (defaults to 1.)

Note that the result preserves the same number of dimensions as the input.
To drop the dimension (dims) that is summed over, use `logsumexpdd`.
"""
function logsumexp(logx::T; dims::Integer=1) where {T <: AbstractArray}
    d = dims < 0 ? ndims(logx) + 1 + dims : dims
    mx = maximum(logx, dims=d)
    return mx .+ log.(sum(exp.(logx .- mx), dims=d))
end


"""
Returns `dropdims(logsumexp(logx, dims=dims), dims=dims)`.
See: `logsumexp`
"""
function logsumexpdd(logx::T; dims::Integer=1) where {T <: AbstractArray}
    d = dims < 0 ? ndims(logx) + 1 + dims : dims
    return dropdims(logsumexp(logx, dims=d), dims=d)
end


"""
Log pdf of normal distribution. This is can be vectorized and doesn't require 
creating `Distribution.Normal` objects.
"""
function lpdf_normal(x::X, m::M, s::S) where {X <: Real, M <: Real, S<:Real}
    z = (x - m) / s
    return -0.5 * log(2 * pi) - z * z * 0.5 - log(s)
end


"""
log pdf of gaussian mixture model. This can be vectorized and
doesn't require creating a `Distributions.MixtureModels` object.
"""
function lpdf_gmm(x::TX, m::TM, s::TS, w::TW;
                  dims::Integer, dropdim::Bool=false) where {TX, TW, TM, TS}
    @assert all(s .> 0)
    @assert all(w .>= 0)
    if dropdim
        return logsumexpdd(log.(w) .+ lpdf_normal.(x, m, s), dims=dims)
    else
        return logsumexp(log.(w) .+ lpdf_normal.(x, m, s), dims=dims)
    end
end


@doc raw"""
Stick-breaking function.

Arguments
=========
- `v`: A vector of length $K - 1$, where $K > 1$. 

Return
======
- A simplex (w) of dimension $K$. Where ∑ₖ wₖ = 1, and each wₖ ≥ 0.
"""
function stickbreak(v)
    K = length(v) + 1
    cumprod_one_minus_v = cumprod(1 .- v)

    eta = [if k == 1 
               v[1]
           elseif k == K
               cumprod_one_minus_v[K-1]
           else
               v[k] * cumprod_one_minus_v[k-1]
           end
           for k in 1:K]

    return eta
end


end  # End of module BnpUtil
