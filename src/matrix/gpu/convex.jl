function qnBalancing_double{T<:AbstractFloat}(A::AFArray{T, 2}, ϵ=1.0e-9, max_iter=65535)
    M, N = size(A)

    issym = issymmetric(A)
    _g! = issym ? function(grad, x)
        @afgc exx = exp.(x)
        @afgc grad[:] = exx .* (A * exx) .- 1
        finalize(exx)
    end : function(grad, x)
        @afgc exx = exp.(x)
        @afgc grad[1:M]     = exx[1:M].*(A*exx[M+1:M+N])
        @afgc grad[M+1:M+N] = exx[M+1:M+N].*(A'*exx[1:M])
        @afgc grad -= 1
        finalize(exx)
    end
    g! = !log_norm ? _g! : 
        function(grad, x)
            _g!(grad, x)
            @printf "norm=%.13f\n" norm(grad)
        end

    grad = AFArray(issym ? zeros(N) : zeros(M+N))
    grad_p = copy(grad)
    sbuf = []
    ybuf = []

    function tloop(g, ss, ys)
        p = -g
        local m = length(ss)
        if (m == 0) return p end
        alphas = zeros(m)
        for i = m:-1:1
            @afgc alphas[i] = (ss[i]' * p) / (ss[i]' * ys[i])
            @afgc p .-= alphas[i]*ys[i]
        end
        @afgc p *= (ss[end]' * ys[end]) / (ys[end]' * ys[end])
        for i=1:m
            @afgc beta = (ys[i]' * p) / (ss[i]' * ys[i])
            @afgc p += (alphas[i]-beta)ss[i]
        end
        return p
    end

    x = AFArray(issym ? -log.(sum(A, 1)[1,:]) ./ 2 :
        -log.(vcat(sum(A, 2)[:,1], sum(A, 1)[1,:])) ./ 2)
    # x = issym ? zeros(N) : zeros(M+N)

    m=10
    k=1
    g!(grad, x)
    ϵ = issym ? ϵ/sqrt(2) : ϵ
    while norm(grad) > ϵ
        p = tloop(grad, sbuf, ybuf)
        @afgc x += p
        grad_p .= grad
        g!(grad, x)
        if k > m
            _s, sbuf = sbuf[1], sbuf[2:end]
            _y, ybuf = ybuf[1], ybuf[2:end]
            finalize(_s)
            finalize(_y)
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
end

