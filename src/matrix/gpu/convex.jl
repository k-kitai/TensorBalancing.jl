function qnBalancing_double{T<:AbstractFloat}(A::AFArray{T, 2}, ϵ=1.0e-9, max_iter=65535; log_norm=false, only_x=false, issym::Bool=true)
    M, N = size(A)
    
    # issym = issymmetric(A)
    _g! = issym ? function(grad, x)
        exx = exp.(x)
        grad[:] = exx .* (A * exx) .- 1
    end : function(grad, x)
        exx = exp.(x)
        grad[1:M]     = exx[1:M].*(A*exx[M+1:M+N])
        grad[M+1:M+N] = exx[M+1:M+N].*(A'*exx[1:M])
        grad -= 1
    end
    g! = !log_norm ? _g! : 
        function(grad, x)
            _g!(grad, x)
        end
    
    grad = AFArray(issym ? zeros(N) : zeros(M+N))
    grad_p = AFArray(issym ? zeros(N) : zeros(M+N))
    sbuf = []
    ybuf = []
    
    function tloop(g, ss, ys)
        p = -g
        local m = length(ss)
        if (m == 0) return p end
        alphas = AFArray(zeros(m))
        for i = m:-1:1
            alphas[i] = (ss[i]' * p) / (ss[i]' * ys[i])
            p .-= alphas[i]*ys[i]
        end
        p *= (ss[end]' * ys[end]) / (ys[end]' * ys[end])
        for i=1:m
            beta = (ys[i]' * p) / (ss[i]' * ys[i])
            p += (alphas[i]-beta)[1] * ss[i]
        end
        return p
    end
    
    x = AFArray(issym ? zeros(N) : zeros(M+N))
    exxx = AFArray(issym ? zeros(N) : zeros(M+N))
    if issym 
        x .= -log.(A*AFArray(ones(N))) / 2
    else
        x[1:M]     = -log.(A *AFArray(ones(N))) / 2
        x[M+1:M+N] = -log.(A'*AFArray(ones(M))) / 2
    end
    # x = issym ? zeros(N) : zeros(M+N)
    
    m=10
    k=1
    _g!(grad, x)
    ϵ = issym ? ϵ/sqrt(2) : ϵ
    while norm(grad) > ϵ
        p = tloop(grad, sbuf, ybuf)
        x += p
        grad_p .= grad
        _g!(grad, x)
        if k > m
            _s, sbuf = sbuf[1], sbuf[2:end]
            _y, ybuf = ybuf[1], ybuf[2:end]
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
