using Logging

"""
    nBalancing{T<:AbstractArray}(A::Matrix{T})

Matrix balancing algorithm based on information geometry
and Newton's Method.
"""
function nBalancing{T<:AbstractFloat}(A::Matrix{T}, ϵ=1.0e-9, max_iter=NaN)
    M, N = size(A)
    targetη = genTargetη(A)
    Δθ = zeros(M+N-2) # scaling factor of each row and column
    ul_index = min.(cumsum(ones(Int64, M-1, M-1), 1), cumsum(ones(Int64, M-1, M-1), 2))
    lr_index = min.(cumsum(ones(Int64, N-1, N-1), 1), cumsum(ones(Int64, N-1, N-1), 2)) .+ (M-1)
    P = copy(A)

    α = 1.0
    r = 1.0 - 1 / (M+N)
    counter = 0
    while(calcRes(P) > ϵ)
        η = mat2η(P)
        innη = η[1:M-1, 1:N-1]
        objη = vcat(η[1:M-1, N], η[M, 1:N-1])
        grad = objη - targetη
        J = objη * objη'
        J[1:M-1,   1:M-1]   -= objη[ul_index]
        J[M:M+N-2, 1:M-1]   -= innη'
        J[1:M-1,   M:M+N-2] -= innη
        J[M:M+N-2, M:M+N-2] -= objη[lr_index]
        α *= r
        try
            Δθ += (J \ grad) .* (1 - α)
        catch e
            if (typeof(e) == Base.LinAlg.SingularException)
                err("The Hessian got singular at $counter'th cycle. Input matrix may be too sparse or ϵ be too small.")
                break
            else
                throw(err)
            end
        end
        P = applyΔθ(A, Δθ)
        counter += 1
        if (max_iter <= counter)
            break
        end
    end
    P
end