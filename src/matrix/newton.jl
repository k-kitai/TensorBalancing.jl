using Logging

"""
    nBalancing{T<:AbstractArray}(A::Matrix{T})

Matrix balancing algorithm based on information geometry
and Newton's Method.
"""
function nBalancing{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN, norm_check=false)
    applyΔθ(A, _nBalancing(A, ϵ, max_iter, norm_check))
end

function _nBalancing{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN, norm_check=false)
    M, N = size(A)
    targetη = genTargetη(A)
    Δθ = zeros(M+N-2) # scaling factor of each row and column
    ul_index = min.(cumsum(ones(Int64, M-1, M-1), 1), cumsum(ones(Int64, M-1, M-1), 2))
    lr_index = min.(cumsum(ones(Int64, N-1, N-1), 1), cumsum(ones(Int64, N-1, N-1), 2)) .+ (M-1)
    P = copy(A)

    error_handler(e::Base.LinAlg.SingularException) = err("The Hessian got singular at $counter'th cycle. Input matrix may be too sparse or ϵ be too small.")
    error_handler(e::Exception) = throw(e)

    α = 1.0
    r = 1.0 - 1 / (M+N)
    counter = 0
    prev_norm = Inf
    residual = calcRes(P)
    while residual > ϵ && (isnan(max_iter) || counter < max_iter)
        η = mat2η(P)
        innη = η[1:M-1, 1:N-1]
        objη = vcat(η[1:M-1, N], η[M, 1:N-1])
        grad = objη - targetη
        J = objη * objη'
        J[1:M-1,   1:M-1]   -= objη[ul_index]
        J[M:M+N-2, 1:M-1]   -= innη'
        J[1:M-1,   M:M+N-2] -= innη
        J[M:M+N-2, M:M+N-2] -= objη[lr_index]
        try
            δ = J \ grad
        catch e
            error_handler(e)
            break
        end
        # If jacobian is nearly singular, δ could be too large step instead of an error being raised.
        if norm_check
            curr_norm = norm(δ)
            if curr_norm > prev_norm * 1.5
                δ *= prev_norm / curr_norm
            else
                prev_norm = curr_norm
            end
        end
        α *= r
        Δθ += δ .* (1 - α)
        P = applyΔθ(A, Δθ)
        residual = calcRes(P)
        counter += 1
    end
    Δθ
end

"""
    recBalancing{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN)

"""
recBalancing{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN, depth=3) = applyΔθ(A, _recBalancing(A, ϵ, max_iter))
function _recBalancing{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN, max_depth=2, depth=0)
    M, N = size(A)
    M1 = div(M, 2)
    M2 = M - M1
    N1 = div(N, 2)
    N2 = N - N1

    Δθ = zeros(M+N-2) # scaling factor of each row and column
    if (depth < max_depth && M > 16 && N > 16)
        Δθ1 = _recBalancing(A[1:N1, 1:M1], ϵ * 2.0, max_iter, max_depth, depth+1)
        Δθ2 = _recBalancing(A[N1+1:N1+N2, M1+1:M1+M2], ϵ * 2.0, max_iter, max_depth, depth+1)
        Δθ[1:M1-1]     = Δθ1[1:M1-1]     # length M1-1   = M1-1
        Δθ[M:M+N1-2]   = Δθ1[M1:M1+N1-2] # length N1-1   = N1-1
        Δθ[M1+1:M-1]   = Δθ2[1:M2-1]     # length M-M1-1 = M2-1
        Δθ[M+N1:M+N-2] = Δθ2[M2:M2+N2-2] # length N-N1-1 = N2-1
        P = applyΔθ(A, Δθ)
        ul_sum = sum(P[1:M1, 1:N1])
        lr_sum = sum(P[M1+1:M, N1+1:N])
        block_scaling = log(lr_sum / ul_sum) / 2
        Δθ[M1] += block_scaling
        Δθ[M+N1-1] += block_scaling
    end
    Δθ += _nBalancing(applyΔθ(A, Δθ), ϵ, max_iter)
    Δθ
end

#============ GPU version ==============#
using ArrayFire

"""
    nBalancing_gpu{T<:AbstractArray}(A::Matrix{T})

Matrix balancing algorithm based on information geometry
and Newton's Method.
"""
function nBalancing_gpu{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9)
    M, N = size(A)
    targetη = genTargetη(A)
    Δθ = zeros(M+N-2) # scaling factor of each row and column
    ul_index = min.(cumsum(ones(Int64, M-1, M-1), 1), cumsum(ones(Int64, M-1, M-1), 2))
    lr_index = min.(cumsum(ones(Int64, N-1, N-1), 1), cumsum(ones(Int64, N-1, N-1), 2)) .+ (M-1)
    P = copy(A)
    α = 1.0
    while(calcRes(P) > ϵ)
        η = Array(mat2η(AFArray(P)))
        innη = η[1:M-1, 1:N-1]
        objη = vcat(η[1:M-1, N], η[M, 1:N-1])
        grad = objη - targetη
        J = objη * objη'
        J[1:M-1,   1:M-1]   .-= objη[ul_index]
        J[M:M+N-2, 1:M-1]   .-= innη'
        J[1:M-1,   M:M+N-2] .-= innη
        J[M:M+N-2, M:M+N-2] .-= objη[lr_index]

        afJ = AFArray(J)
        piv = lu_inplace(afJ, true)
        α *= 0.99
        Δθ += Array(solve_lu(afJ, piv, AFArray(grad), AF_MAT_NONE)) .* (1 - α)

        P = applyΔθ(A, Δθ)
    end
    P
end
