function qnBalancing{T<:AbstractFloat}(A::AFArray{T, 2}, ϵ=1.0e-9, max_iter=65535)
    M, N = size(A)

    function f(x) 
        @afgc v = sum(log.(A*exp.(AFArray(x)))) - sum(x)
        return v
    end
    function g!(grad, x)
        @afgc exx = exp.(AFArray(x))
        @afgc rowsums_inv = 1./(A * exx)
        @afgc grad[:] = squeeze(rowsums_inv' * A, 1) .* exx .- 1
        finalize(exx)
        finalize(rowsums_inv)
    end

    initialX = -log.(Base.squeeze(sum(Array(A), 1), 1))

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
    P = Array(A) .* exp.(result.minimizer)'
    P ./ sum(P, 2)
end

