# Matrix Balancing Methods based on fixed-point iteration
function skBalancing{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN)
    if issymmetric(A)
        _skBalancing_sym(A, ϵ, max_iter)
    else
        _skBalancing_nosym(A, ϵ, max_iter)
    end
end

function _skBalancing_nosym{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN)
    M, N = size(A)
    r = ones(M)
    c = ones(N)
    counter = 0
    residual = calcRes(A .* r .* c')
    while residual > ϵ && (isnan(max_iter) || counter < max_iter)
        r .= 1 ./ (A * c)
        c .= 1 ./ (A' * r)
        residual = calcRes(A .* r .* c')
        counter += 1
        # @show residual
    end
    r, c
end

function _skBalancing_sym{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN)
    @assert issymmetric(A) "Input matrix should be symmetric."
    N, _ = size(A)
    x = ones(N)
    counter = 0
    residual = norm(x.*(A * x))
    ϵ /= sqrt(2) # correction for comparison
    while residual > ϵ && (isnan(max_iter) || counter < max_iter)
        x .= 1 ./ (A * x)
        residual = norm(x.*(A * x))
        counter += 1
        # @show residual
    end
    x
end

"""
    fpnBalancing{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN)

Fixed point iteration with Newton's method
"""
function fpnBalancing{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN)
    @assert issymmetric(A) "Input matrix should be symmetric."
    N, _ = size(A)
    x = ones(N)
    counter = 0
    residual = calcRes(A .* x .* x')
    α = copy(A)
    while residual > ϵ && (isnan(max_iter) || counter < max_iter)
        Ax = A * x
        for i = 1:N
            α[i,i] = A[i,i] + Ax[i] / x[i]
            # α[i,i] = A[i,i] + 1 / x[i] / x[i]
        end
        x .= (A .* x + diagm(Ax)) \ (Ax .* x .+ 1)
        residual = calcRes(A .* x .* x')
        counter += 1
    end
    x
end
