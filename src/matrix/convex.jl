using Optim
using Logging
using NLSolversBase
using LineSearches
using Base.Threads

# struct QN_WRAPPER <: Optimizer
#     lbfgs::LBFGS
# end
# initial_state(method::QN_WRAPPER, args...) = Optim.initial_state(method.lbfgs, args...)
# update_state!(d, state::LBFGSState{T}, method::QN_WRAPPER) = Optim.update_state!(d, state, method.lbfgs)
# update_h!(d, state, method::QN_WRAPPER) = Optim.update_h!(d, state, method.lbfgs)
# assess_convergence(state, d, options)

"""
    qnBalancing{T<:AbstractArray}(A::Matrix{T})

Matrix balancing algorithm based using LBFGS
"""

function qnBalancing{T<:AbstractFloat}(A::AbstractArray{T, 2}, ϵ=1.0e-9, max_iter=65535; log_norm=false, only_x=false)
    M, N = size(A)

    function _g!(grad, x)
        exx = exp.(x)
        rowsums_inv = 1./ (A * exx)
        grad .= Base.squeeze(rowsums_inv' * A, 1) .* exx - 1
    end
    g! = !log_norm ? _g! : 
        function(grad, x)
            _g!(grad, x)
            @printf "norm=%.18f\n" norm(grad)
        end

    grad = zeros(N)
    grad_p = zeros(N)
    sbuf = []
    ybuf = []

    function tloop(g, ss, ys)
        p = -g
        local m = length(ss)
        if (m == 0) return p end
        alphas = zeros(m)
        for i = m:-1:1
            alphas[i] = (ss[i]' * p) / (ss[i]' * ys[i])
            p .-= alphas[i]*ys[i]
        end
        p *= (ss[end]' * ys[end]) / (ys[end]' * ys[end])
        for i=1:m
            beta = (ys[i]' * p) / (ss[i]' * ys[i])
            p += (alphas[i]-beta)ss[i]
        end
        return p
    end

    x = -log.(A*ones(N))[:,1]

    m=10
    k=1
    g!(grad, x)
    # ϵ = ϵ/sqrt(2)
    while norm(grad) > ϵ && k <= max_iter
        p = tloop(grad, sbuf, ybuf)
        x += p
        grad_p .= grad
        g!(grad, x)
        if k > m
            sbuf = sbuf[2:end]
            ybuf = ybuf[2:end]
        end
        push!(sbuf, p)
        push!(ybuf, grad - grad_p)
        k += 1
    end

    if only_x
        return x
    end
    P = A .* exp.(x)'
    P ./ sum(P, 2)
end

if USE_AF
    include("gpu/convex.jl")
end #USE_AF

function qnBalancing_double{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=65535; log_norm=false, only_x=false)
    M, N = size(A)

    issym = issymmetric(A)
    _g! = issym ? function(grad, x)
        exx = exp.(x)
        grad .= exx .* (A * exx) .- 1
    end : function(grad, x)
        exx = exp.(x)
        grad .= vcat(exx[1:M].*(A*exx[M+1:M+N]), exx[M+1:M+N].*(A'*exx[1:M])) .- 1
    end
    g! = !log_norm ? _g! : 
        function(grad, x)
            _g!(grad, x)
            @printf "norm=%.18f\n" norm(grad)*(issym?sqrt(2):1)
        end

    grad = issym ? zeros(N) : zeros(M+N)
    grad_p = copy(grad)
    sbuf = []
    ybuf = []

    function tloop(g, ss, ys)
        p = -g
        local m = length(ss)
        if (m == 0) return p end
        alphas = zeros(m)
        for i = m:-1:1
            alphas[i] = (ss[i]' * p) / (ss[i]' * ys[i])
            p .-= alphas[i]*ys[i]
        end
        p *= (ss[end]' * ys[end]) / (ys[end]' * ys[end])
        for i=1:m
            beta = (ys[i]' * p) / (ss[i]' * ys[i])
            p += (alphas[i]-beta)ss[i]
        end
        return p
    end

    x = issym ? -log.(sum(A, 1)[1,:]) ./ 2 :
        -log.(vcat(sum(A, 2)[:,1], sum(A, 1)[1,:])) ./ 2
    # x = issym ? zeros(N) : zeros(M+N)

    m=10
    k=1
    g!(grad, x)
    ϵ = issym ? ϵ/sqrt(2) : ϵ
    while norm(grad) > ϵ && k <= max_iter
        p = tloop(grad, sbuf, ybuf)
        x += p
        grad_p .= grad
        g!(grad, x)
        if k > m
            sbuf = sbuf[2:end]
            ybuf = ybuf[2:end]
        end
        push!(sbuf, p)
        push!(ybuf, grad - grad_p)
        k += 1
    end
    # @show result.minimizer
    if only_x
        return x
    end
    exx = exp.(x)
    issym ? A .* exx .* exx' : A .* exx[1:M] .* exx[M+1:M+N]'
end
