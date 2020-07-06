module BnpUtil

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
