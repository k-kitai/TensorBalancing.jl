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

    nth = nthreads()
    blocksize = div(M, nth)
    rowblocks = [blocksize*n+1:blocksize*(n+1) for n = 0:nth-1]
    rowblocks[end] = blocksize*(nth-1)+1 : M
    f(x) = log(prod(A*exp.(x))) - sum(x)
    function _g!(grad, x)
        exx = exp.(x)
        rowsums_inv = 1./ (A * exx)
        grad .= squeeze(rowsums_inv' * A, 1) .* exx .- 1
    end
    g! = !log_norm ? _g! : 
        function(grad, x)
            _g!(grad, x)
            @printf "norm=%.13f\n" norm(grad)
        end

    initialX = -log.(Base.squeeze(sum(A, 1), 1))

    result = optimize((f, g!),
            initialX,
            LBFGS(linesearch = Static(alpha=1.0)),
            # LBFGS(linesearch = HagerZhang(0.1, 0.9, 1.0, 5.0, 1e-6, 0.66, 50, 0.1, 0)),
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
    if only_x
        return result.minimizer
    end
    P = A .* exp.(result.minimizer)'
    P ./ sum(P, 2)
end

if USE_AF
    include("gpu/convex.jl")
end #USE_AF

function qnBalancing_double{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=65535; log_norm=false, only_x=false)
    M, N = size(A)

    issym = issymmetric(A)
    f = issym ? function(x)
        exx = exp.(x)
        exx' * A * exx / 2 - sum(x) + 0.5x[end]
        # The last term is only for preventing iterations from stopping because of f_converged.
        # Its gradient shouldn't be included.
    end : function(x)
        # r, c = x[1:M], x[M+1:M+N]
        exx = exp.(x)
        sum(exx[1:M]' * A * exx[M+1:M+N]) - sum(x) + 0.5x[end]
        # The last term is only for preventing iterations from stopping because of f_converged.
        # Its gradient shouldn't be included.
    end
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
            @printf "norm=%.13f\n" norm(grad)
        end

    initialX = issym ? -log.(Base.squeeze(sum(A, 1), 1)) ./ 2 :
        -log.(vcat(Base.squeeze(sum(A, 2), 2), Base.squeeze(sum(A, 1), 1))) ./ 2
    result = optimize((f, g!),
            initialX,
            LBFGS(linesearch = HagerZhang(0.1, 0.9, 1.0, 5.0, 1e-6, 0.66, 50, 0.1, 0)),
            Optim.Options(
                x_tol=-1.0,
                f_tol=-1.0,
                g_tol=ϵ/sqrt(N+M),
                iterations=max_iter,
                show_trace=false,
                show_every=1,
                allow_f_increases=true
            ))
    # @show result.minimizer
    if only_x
        return result.minimizer
    end
    exx = exp.(result.minimizer)
    issym ? A .* exx .* exx' : A .* exx[1:M] .* exx[M+1:M+N]'
end
