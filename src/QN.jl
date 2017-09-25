# QN.jl
# Copyright 2017 Koki Kitai.
# 
# Quasi-Newton methods
using Optim
using LineSearches
include("lnsrch.jl")

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
function qnBalancing{T<:AbstractFloat}(P::Array{T,2}; ϵ=1e-6, max_iter=1000, show_trace=false, show_every=100, limited=false)
    A = copy(P)
    ylen, xlen = size(A)
    cumθ_y, cumθ_x = zeros(ylen), zeros(xlen)
    η = zeros(xlen+ylen-1)
    β = genβ(A::Array{T,2})
    β = β[2:end]
    η_target = genTargetη(A, β)

    function f(θ)
        cumθ!(cumθ_x, cumθ_y, θ, β, size(A))
        local s = 0.0
        for j = 1:xlen, i = 1:ylen
            if P[i,j] > 1e-15 && A[i,j] > 1e-100
                A[i,j] = P[i,j] * exp(cumθ_y[i] + cumθ_x[j])
                if A[i,j] <= 1e-100
                    A[i,j] = 0.0
                end
                s += A[i,j]
            end
        end
        log(s) - η_target'θ
    end

    function g!(grad, θ)
        calcMarginals!(η, A)
        η ./= η[1]
        for i in 1:length(β)
            grad[i] = η[i+1] - η_target[i]
        end
    end

    function cb(x)
        local res = calcResidual(A, xlen/sum(A))
        return  res < ϵ
    end

    result = 0
    result = optimize((f, g!),
            zeros(length(β)), # initial θ
            if limited
                LBFGS(linesearch=Lnsrch(0.5, .0, 20, f), m=100)
                #LBFGS(linesearch=LineSearches.BackTracking(), m=100)
            else
                BFGS(linesearch=Lnsrch(0.5, .0, 10, f))
                #BFGS(linesearch=LineSearches.BackTracking())
            end,
            Optim.Options(
                    x_tol=NaN,
                    f_tol=NaN,
                    g_tol=NaN,
                    iterations=max_iter,
                    show_trace=show_trace,
                    show_every=show_every,
                    allow_f_increases=true,
                    callback=cb,
            ))
    return A./sum(A)
end

"""
    ltqnBalancing{T<:AbstractFloat}(P::Array{T,2}; ϵ=1e-6, max_iter=1000, show_trace=false, show_every=100)

Matrix balancing by E-projection with Quasi-Newton method on a linear-transformed space.

# Parameters
* P - Input matrix
* ϵ - Threshold for the error of η-coordinate
* max_iter - Max number of iterations for optimization
* show_trace - Whether to print trace
* show_every - Trace information is printed every `show_every`'th iteration
"""
function ltqnBalancing{T<:AbstractFloat}(P::Array{T,2}; ϵ=1e-6, max_iter=1000, show_trace=false, show_every=100, limited=false)
    A = copy(P)
    ylen, xlen = size(A)
    cumθ_y, cumθ_x = zeros(ylen), zeros(xlen)
    T_inv = eye(xlen+ylen-2)
    for i = 2:xlen+ylen-2
        T_inv[i,i-1] = -1.0
    end
    T_inv[xlen, xlen-1] = 0.0

    θ_y, θ_x = zeros(ylen), zeros(xlen)
    η = zeros(xlen+ylen-1)
    β = genβ(A::Array{T,2})
    β = β[2:end]
    η_target = genTargetη(A, β)

    function f(θ)
        cumθ!(cumθ_x, cumθ_y, θ, β, size(A))
        local s = 0.0
        for j = 1:xlen, i = 1:ylen
            fact  = 1 == i ? 0 : θ[i-1]
            fact += 1 == j ? 0 : θ[xlen-2+j]
            if P[i,j] > 1e-15 && A[i,j] > 1e-100
                A[i,j] = P[i,j] * exp(fact)
                #A[i,j] = P[i,j] * exp(cumθ_y[i] + cumθ_x[j])
                if A[i,j] <= 1e-100
                    A[i,j] = 0.0
                end
                s += A[i,j]
            end
        end
        log(s) - η_target' * (T_inv * θ)
        #log(s) - η_target' * θ
    end

    function g!(grad, θ)
        calcMarginals!(η, A)
        η ./= η[1]
        grad .= T_inv * (η[2:end] - η_target)
        #grad .= η[2:end] - η_target
    end

    function cb(x)
        local res = calcResidual(A, xlen/sum(A))
        print(A)
        #print("res: $res\n")
        return  res < ϵ
    end

    result = 0
    result = optimize((f, g!),
            zeros(length(β)), # initial θ
            if limited
                LBFGS(linesearch=Lnsrch(0.5, .0, 30, f), m=100)
                #LBFGS(linesearch=LineSearches.BackTracking(), m=100)
            else
                BFGS(linesearch=Lnsrch(0.5, .0, 30, f))
                #BFGS(linesearch=LineSearches.BackTracking())
            end,
            Optim.Options(
                    x_tol=NaN,
                    f_tol=NaN,
                    g_tol=NaN,
                    iterations=max_iter,
                    show_trace=show_trace,
                    show_every=show_every,
                    allow_f_increases=true,
                    callback=cb,
            ))
    return A./sum(A)
end

