"""
    calcRes
"""
function calcRes{T<:AbstractFloat}(A::Matrix{T})
    M, N = size(A)
    colsum = sum(A, 1)
    rowsum = sum(A, 2)
    allsum = sum(colsum)
    norm(vcat(colsum' .* N, rowsum .* M) ./ allsum  .- 1.0)
end

#============ GPU implementation ==============#

"""
    calcRes
"""
function calcRes{T<:AbstractFloat}(A::AFArray{T})
    M, N = size(A)
    colsum = sum(A, 1)
    rowsum = sum(A, 2)
    allsum = sum(colsum)
    norm(vcat(colsum' .* N, rowsum .* M) ./ allsum  .- T(1.0))
end
