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

    # hotstarter = Base.squeeze(1 ./ sum(A, 1), 1)
    lower = ones(N) * (-400.0)
    upper = ones(N) * 400.0
    initial_x = zeros(N)
    od = OnceDifferentiable(f, g!, initial_x)
    optimizer = Fminbox{BFGS}()
    result = optimize(od,
            initial_x,
            lower, upper,
            optimizer,
            g_tol=ϵ/sqrt(N),
            iterations=max_iter,
            show_trace=false,
            show_every=1,
            allow_f_increases=true)
    # result = optimize((f, g!),
    #         zeros(N),
    #         BFGS(linesearch = HagerZhang(0.1, 0.9, 1.0, 5.0, 1e-6, 0.66, 50, 0.1, 0)),
    #         Optim.Options(
    #             x_tol=-1.0,
    #             f_tol=-1.0,
    #             g_tol=ϵ/sqrt(N),
    #             iterations=max_iter,
    #             show_trace=false,
    #             show_every=1,
    #             allow_f_increases=true
    #         ))
    # @show result.minimizer
    P = A .* exp.(result.minimizer)'
    P ./ sum(P, 2)
end

function qnBalancing_double{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=65535)
    M, N = size(A)

    function f(x)
        r, c = x[1:M], x[M+1:M+N]
        sum(A .* exp.(r .+ c')) - sum(r) - sum(c)
    end
    function g!(grad, x)
        r, c = x[1:M], x[M+1:M+N]
        P = A .* exp.(r .+ c')
        grad .= Base.squeeze(vcat(sum(P, 2), sum(P, 1)'), 2) .- 1
    end

    lower = ones(M+N) * (-400.0)
    upper = ones(M+N) * 400.0
    initial_x = zeros(M+N)
    od = OnceDifferentiable(f, g!, initial_x)
    optimizer = Fminbox{BFGS}()
    result = optimize(od,
            initial_x,
            lower, upper,
            optimizer,
            x_tol=-1.0,
            f_tol=-1.0,
            g_tol=ϵ/sqrt(M+N),
            iterations=max_iter,
            show_trace=false,
            show_every=5,
            linesearch = HagerZhang(0.1, 0.9, 1.0, 5.0, 1e-6, 0.66, 50, 0.1, 0),
            allow_f_increases=true)
    # result = optimize((f, g!),
    #         zeros(M+N),
    #         BFGS(linesearch = HagerZhang(0.1, 0.9, 1.0, 5.0, 1e-6, 0.66, 50, 0.1, 0)),
    #         Optim.Options(
    #             x_tol=-1.0,
    #             f_tol=-1.0,
    #             g_tol=ϵ/sqrt(N),
    #             iterations=max_iter,
    #             show_trace=true,
    #             show_every=1,
    #             allow_f_increases=true
    #         ))
    @show result.minimizer
    r, c = result.minimizer[1:M], result.minimizer[M+1:M+N]
    A .* exp.(r .+ c')
end
