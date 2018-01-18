using Optim
using Logging
using NLSolversBase
using LineSearches

"""
    qnBalancing{T<:AbstractArray}(A::Matrix{T})

Matrix balancing algorithm based using LBFGS
"""

function qnBalancing{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=65535)
    M, N = size(A)

    f(x) = sum(log.(A*exp.(x))) - sum(x)
    function g!(grad, x)
        grad .= squeeze(sum((A .* exp.(x')) ./ (A * exp.(x)), 1) .- 1, 1)
    end

    initialX = -log.(Base.squeeze(sum(A, 1), 1))

    result = optimize((f, g!),
            initialX,
            BFGS(linesearch = HagerZhang(0.1, 0.9, 1.0, 5.0, 1e-6, 0.66, 50, 0.1, 0)),
            Optim.Options(
                x_tol=-1.0,
                f_tol=-1.0,
                g_tol=ϵ/sqrt(N),
                iterations=max_iter,
                show_trace=false,
                show_every=1,
                allow_f_increases=true
            ))
    # @show result.minimizer
    P = A .* exp.(result.minimizer)'
    P ./ sum(P, 2)
end

function qnBalancing_double{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=65535)
    M, N = size(A)

    issym = issymmetric(A)
    f = issym ? function(x)
        sum(A .* exp.(x .+ x'))/2 - sum(x)
    end : function(x)
        r, c = x[1:M], x[M+1:M+N]
        sum(A .* exp.(r .+ c')) - sum(r) - sum(c)
    end
    g! = issym ? function(grad, x)
        grad .= Base.squeeze(sum(A .* exp.(x .+ x'), 2), 2) .- 1
    end : function(grad, x)
        r, c = x[1:M], x[M+1:M+N]
        P = A .* exp.(r .+ c')
        grad .= Base.squeeze(vcat(sum(P, 2), sum(P, 1)'), 2) .- 1
    end

    initialX = issym ? -log.(Base.squeeze(sum(A, 1), 1)) ./ 2 :
        -log.(vcat(Base.squeeze(sum(A, 2), 2), Base.squeeze(sum(A, 1), 1))) ./ 2
    result = optimize((f, g!),
            initialX,
            LBFGS(linesearch = HagerZhang(0.1, 0.9, 1.0, 5.0, 1e-6, 0.66, 50, 0.1, 0)),
            Optim.Options(
                x_tol=0.0,
                f_tol=0.0,
                g_tol=ϵ/sqrt(N),
                iterations=max_iter,
                show_trace=false,
                show_every=1,
                allow_f_increases=true
            ))
    # @show result.minimizer
    if issym
        return A .* exp.(result.minimizer .+ result.minimizer')
    else
        r, c = result.minimizer[1:M], result.minimizer[M+1:M+N]
        A .* exp.(r .+ c')
    end
end
