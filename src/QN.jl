# QN.jl
# Copyright 2017 Koki Kitai.
# 
# Quasi-Newton methods
using Optim

"""
    qnBalancing{T<:AbstractFloat}(P::Array{T,2})

Matrix balancing by E-projection with LBFGS.

# Parameters
* P - Input matrix
"""
function qnBalancing{T<:AbstractFloat}(P::Array{T,2})
    A = copy(P)
    ylen, xlen = size(A)
    cumθ_y, cumθ_x = zeros(ylen), zeros(xlen)
    η = zeros(xlen+ylen-1)
    β = genβ(A::Array{T,2})
    η_target = genTargetη(A, β)

    function f(θ)
        cumθ!(cumθ_x, cumθ_y, θ, β, size(A))
        s = 0.0
        for j = 1:xlen, i = 1:ylen
            if P[i,j] > 1e-15
                A[i,j] = P[i,j] * exp(cumθ_y[i] + cumθ_x[j])
                s += A[i,j]
            end
        end
        log(s) - (η_target'θ)[1]
    end

    function g!(grad, θ)
        calcMarginals!(η, A)
        η ./= η[1]
        for i in 1:length(β)
            grad[i] = η[i] - η_target[i]
        end
    end

    result = optimize(f, g!, zeros(length(β)), LBFGS())

    f(result.minimizer)
    return A./sum(A)
end
