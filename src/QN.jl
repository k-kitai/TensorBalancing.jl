# QN.jl
# Copyright 2017 Koki Kitai.
# 
# Quasi-Newton methods
using Optim
using LineSearches

"""
    qnBalancing{T<:AbstractFloat}(P::Array{T,2}; ϵ=1e-6, max_iter=1000, show_trace=false, show_every=100)

Matrix balancing by E-projection with LBFGS.

# Parameters
* P - Input matrix
* ϵ - Threshold for the error of η-coordinate
* max_iter - Max number of iterations for optimization
* show_trace - Whether to print trace
* show_every - Trace information is printed every `show_every`'th iteration
"""
function qnBalancing{T<:AbstractFloat}(P::Array{T,2}; ϵ=1e-6, max_iter=1000, show_trace=false, show_every=100)
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
        log(s) - η_target'θ
    end

    function g!(grad, θ)
        calcMarginals!(η, A)
        η ./= η[1]
        for i in 1:length(β)
            grad[i] = η[i] - η_target[i]
        end
    end

    result = optimize(f, g!,
            zeros(length(β)),
            LBFGS(linesearch=LineSearches.BackTracking(
                    c1 = 1e-4,
                    rhohi = 0.5,
                    rholo = 0.1,
                    iterations = 1_000,
                    order = 3,
                    maxstep = Inf)),
            Optim.Options(
                    x_tol=NaN,
                    f_tol=NaN,
                    g_tol=ϵ,
                    iterations=max_iter,
                    show_trace=show_trace,
                    show_every=show_every,
                    allow_f_increases=true
            ))

    f(result.minimizer)
    return A./sum(A)
end
