"""
    calcRes
"""
function calcRes{T<:AbstractFloat}(A::Matrix{T})
    M, N = size(A)
    calcRes(A, ones(M), ones(N))
end

function calcRes{T<:AbstractFloat}(A::Matrix{T}, target_row, target_col)
    colsum = squeeze(sum(A, 1), 1)
    rowsum = squeeze(sum(A, 2), 2)
    norm(vcat(colsum - target_col, rowsum - target_row))
end

#============ GPU implementation ==============#

if USE_AF

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

end #USE_AF
