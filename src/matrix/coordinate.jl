"""
    applyΔθ
"""
function applyΔθ{T<:AbstractFloat}(A::Matrix{T}, Δθ::Vector{T})
    cumsum_backward(v) = reverse(cumsum(reverse(v)))

    M, N = size(A)
    rowlogscale = zeros(M)
    collogscale = zeros(N)
    rowlogscale[1:M-1] = cumsum_backward(Δθ[1:M-1])
    collogscale[1:N-1] = cumsum_backward(Δθ[M:N+M-2])
    T.(A .* exp.(min.(rowlogscale .+ collogscale', 500)))
end

"""
    genTargetη{T<:AbstractFloat}(A::Matrix{T})

Generate the target value of η's marginal part.
"""
function genTargetη{T<:AbstractFloat}(A::Matrix{T})
    M, N = size(A)
    T.(vcat((1:M-1)./M, (1:N-1)./N))
end

"""
    mat2η{F<:AbstractFloat, T<:AbstractArray{F,2}}(A::T)

Calculate η coordinate of a matrix.
"""
mat2η{F<:AbstractFloat, T<:AbstractArray{F,2}}(A::T) = cumsum(cumsum(A, 1), 2) ./ sum(A)

