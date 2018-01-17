using Logging
using NLSolversBase
using LineSearches

"""
    nBalancing{T<:AbstractArray}(A::Matrix{T})

Matrix balancing algorithm based on information geometry
and Newton's Method.
"""
function nBalancing{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN)
    M, N = size(A)
    initialΔθ = issymmetric(A) ? zeros(T, M-1) : zeros(T, M+N-2)
    applyΔθ(A, _nBalancing(A, initialΔθ, ϵ, max_iter))
end

function _nBalancing{T<:AbstractFloat}(A::Matrix{T}, initialΔθ, ϵ=1.0e-9, max_iter=NaN)
    M, N = size(A)
    issym = issymmetric(A)
    targetη = issym ? 2genTargetη(A)[1:M-1] : genTargetη(A)
    Δθ = copy(initialΔθ) # scaling factor of each row and column
    Δθtmp = issym ? zeros(T, M-1) : zeros(T, M+N-2)
    δ = issym ? zeros(T, M-1) : zeros(T, M+N-2)
    ul_index = min.(cumsum(ones(Int64, M-1, M-1), 1), cumsum(ones(Int64, M-1, M-1), 2))
    lr_index = min.(cumsum(ones(Int64, N-1, N-1), 1), cumsum(ones(Int64, N-1, N-1), 2)) .+ (M-1)
    P = applyΔθ(A, Δθ)

    error_handler(e::Base.LinAlg.SingularException) = err("The Hessian got singular at $counter'th cycle. Input matrix may be too sparse or ϵ be too small.")
    error_handler(e::Exception) = throw(e)

    f(Δθ) = log(sum(applyΔθ(A, Δθ))) - dot(targetη, Δθ)
    g! = issym ?  function (out, Δθ)
            η = cumsum(sum(applyΔθ(A, Δθ), 2))
            objη = η[1:M-1, 1]
            out .= 2objη - targetη
        end : function (out, Δθ)
            η = mat2η(applyΔθ(A, Δθ))
            objη = vcat(η[1:M-1, N], η[M, 1:N-1])
            out .= objη - targetη
        end
    fg! = issym ? function (out, Δθ)
            η = cumsum(sum(applyΔθ(A, Δθ), 2))
            objη = η[1:M-1, 1]
            out .= 2objη - targetη
            log(η[M]) - dot(targetη, Δθ)
        end : function (out, Δθ)
            local P = applyΔθ(A, Δθ)
            η = mat2η(P)
            out .= vcat(η[1:M-1, N], η[M, 1:N-1])
            log(sum(P)) - dot(targetη, Δθ)
        end
    df = OnceDifferentiable(f, g!, fg!, Δθ)

    counter = 0
    residual = calcRes(P)
    hz = HagerZhang(0.1, 0.9, 1.0, 5.0, 1e-6, 0.66, 50, 0.1, 0) # Line search method
    while residual > ϵ && (isnan(max_iter) || counter < max_iter)
        η = mat2η(P)
        innη = η[1:M-1, 1:N-1]
        objη = issym ? 2η[1:M-1, M] :
                       vcat(η[1:M-1, N], η[M, 1:N-1])
        grad = objη - targetη
        J = objη * objη'
        if issym
            J -= 2innη
            J -= objη[ul_index]
        else
            J[1:M-1,   1:M-1]   -= objη[ul_index]
            J[M:M+N-2, 1:M-1]   -= innη'
            J[1:M-1,   M:M+N-2] -= innη
            J[M:M+N-2, M:M+N-2] -= objη[lr_index]
        end

        try
            δ .= J \ grad
        catch e
            error_handler(e)
            break
        end

        # Line search
        α = 1.0
        lsr = LineSearchResults(eltype(Δθ))
        push!(lsr, 0.0, f(Δθ), -norm(grad))
        α = hz(df, Δθ, δ, Δθtmp, lsr, α, true)
        if α < 1.0e-10
            info("Dismissing too small step: $α")
            α = 1.0
        end

        Δθ += δ .* α
        P = applyΔθ(A, Δθ)
        residual = calcRes(P)
        counter += 1
        # @show α f(Δθ) residual
    end
    Δθ
end

"""
    recBalancing{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN)

"""
recBalancing{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN, max_depth=3, depth=0) = applyΔθ(A, _recBalancing(A, ϵ, max_iter, max_depth, depth))
function _recBalancing{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN, max_depth=3, depth=0)
    M, N = size(A)
    M1 = div(M, 2)
    N1 = div(N, 2)
    M2 = M - M1
    N2 = N - N1

    issym = issymmetric(A)
    Δθ = issym ? zeros(T, M-1) : zeros(T, M+N-2) # scaling factor of each row and column
    if (depth < max_depth && M > 16 && N > 16)
        Δθ1 = _recBalancing(A[1:M1, 1:N1], ϵ * 1.0, max_iter, max_depth, depth+1)
        Δθ2 = _recBalancing(A[M1+1:M1+M2, N1+1:N1+N2], ϵ * 1.0, max_iter, max_depth, depth+1)
        Δθ[1:M1-1]     = Δθ1[1:M1-1]     # length M1-1   = M1-1
        Δθ[M1+1:M-1]   = Δθ2[1:M2-1]     # length M-M1-1 = M2-1
        if !issym
            Δθ[M:M+N1-2]   = Δθ1[M1:M1+N1-2] # length N1-1   = N1-1
            Δθ[M+N1:M+N-2] = Δθ2[M2:M2+N2-2] # length N-N1-1 = N2-1
        end
        P = applyΔθ(A, Δθ)
        ul_sum = sum(P[1:M1, 1:N1])
        lr_sum = sum(P[M1+1:M, N1+1:N])
        block_scaling = log(lr_sum / ul_sum) / 2
        Δθ[M1] += block_scaling
        if !issym
            Δθ[M+N1-1] += block_scaling
        end
    end
    _nBalancing(A, Δθ, ϵ, max_iter)
end

#============ GPU version ==============#
using ArrayFire

"""
    nBalancing_gpu{T<:AbstractArray}(A::Matrix{T})

Matrix balancing algorithm based on information geometry
and Newton's Method.
"""
function nBalancing_gpu{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN)
    M, N = size(A)
    initialΔθ = zeros(T, M+N-2)
    applyΔθ(A, _nBalancing_gpu(A, initialΔθ, ϵ, max_iter))
end

function _nBalancing_gpu{T<:AbstractFloat}(A::Matrix{T}, initialΔθ, ϵ=1.0e-9, max_iter=NaN)
    M, N = size(A)
    targetη = genTargetη(A)
    Δθ = copy(initialΔθ) # scaling factor of each row and column
    Δθtmp = zeros(T, M+N-2)
    ul_index = min.(cumsum(ones(Int64, M-1, M-1), 1), cumsum(ones(Int64, M-1, M-1), 2))
    lr_index = min.(cumsum(ones(Int64, N-1, N-1), 1), cumsum(ones(Int64, N-1, N-1), 2)) .+ (M-1)
    P = applyΔθ(A, Δθ)

    f(Δθ) = log(sum(applyΔθ(A, Δθ))) - dot(targetη, Δθ)
    function g!(out, Δθ)
        η = mat2η(applyΔθ(A, Δθ))
        objη = vcat(η[1:M-1, N], η[M, 1:N-1])
        out .= objη - targetη
    end
    function fg!(out, Δθ)
        local P = applyΔθ(A, Δθ)
        η = mat2η(P)
        out .= vcat(η[1:M-1, N], η[M, 1:N-1])
        log(sum(P)) - dot(targetη, Δθ)
    end
    df = OnceDifferentiable(f, g!, fg!, Δθ)

    counter = 0
    residual = calcRes(P)
    hz = HagerZhang(0.1, 0.9, 1.0, 5.0, 1e-6, 0.66, 50, 0.1, 0) # Line search method
    while residual > ϵ && (isnan(max_iter) || counter < max_iter)
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
        δ = Array(solve_lu(afJ, piv, AFArray(grad), AF_MAT_NONE))

        # Line search
        α = T(1.0)
        lsr = LineSearchResults(eltype(Δθ))
        push!(lsr, 0.0, f(Δθ), -norm(grad))
        α = hz(df, Δθ, δ, Δθtmp, lsr, α, true)
        if α < 1.0e-10
            info("Dismissing too small step: $α")
            α = 1.0
        end

        Δθ += δ .* α
        P = applyΔθ(A, Δθ)
        residual = calcRes(P)
        counter += 1
        # @show α f(Δθ) residual
    end
    Δθ
end
