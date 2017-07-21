# QN.jl
# Copyright 2017 Koki Kitai.
# 
# Quasi-Newton methods
using Optim

"""
    qnBalancing{T<:AbstractFloat}(P::Array{T,2}; max_iter=100, L=10, log_interval=10)

E-projection by LBFGS.

# parameters
* P - Input array
"""
function qnBalancing{T<:AbstractFloat}(P::Array{T,2})
    A = copy(P)
    ylen, xlen = size(A)
    cumθ_y, cumθ_x = zeros(ylen), zeros(xlen)
    η = zeros(xlen+ylen-1)
    β = vcat([(1,1)], [(1,i) for i = 2:xlen], [(i,1) for i = 2:ylen])
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

function calcMarginals!(η, A)
    ylen, xlen = size(A)
    cumcol = zeros(ylen)
    for i = xlen-1:-1:1
        cumcol += A[:,i+1]
        η[i+1] = sum(cumcol)
    end
    cumcol += A[:,1]
    for i = length(cumcol)-1:-1:1
      cumcol[i] += cumcol[i+1]
    end
    η[xlen+1:end] = cumcol[2:end]
    η[1] = cumcol[1]
end
