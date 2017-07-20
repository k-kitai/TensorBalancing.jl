module TensorBalancing
export nBalancing, shBalancing, gdBalancing

using Logging
using DataStructures

"""
    genβ{T<:AbstractArray}(P::Array{T,2})

Collect indices for β.
"""
function genβ{T<:AbstractFloat}(P::Array{T,2})
    ylen, xlen = size(P)
    return vcat([(1,1)], [(1,i) for i = 2:xlen], [(i,1) for i = 2:ylen])
end

"""
    calcη{T<:AbstractArray}(P::Array{T,2})

Calculate η coordinate of a matrix.
"""
function calcη{T<:AbstractFloat}(P::Array{T,2})
    _shape = size(P)
    η = zeros(_shape)
    idx = [_shape...]
    η[idx...] = P[idx...]
    for i in _shape[1]-1:-1:1
        η[i,_shape[2]] = η[i+1, _shape[2]] + P[i, _shape[2]]
    end
    for j in _shape[2]-1:-1:1
        η[_shape[1], j] = η[_shape[1], j+1] + P[_shape[1], j]
    end
    for i in _shape[1]-1:-1:1
        for j in _shape[2]-1:-1:1
            η[i,j] = η[i+1, j] + η[i, j+1] - η[i+1, j+1] + P[i,j] # de morgan
        end
    end
    return η
end

"""
    genTargetη(P::Array{T}, β)

Generate target value of elements in η.
"""
function genTargetη{T<:AbstractFloat}(P::AbstractArray{T}, β)
    shape = size(P)
    dim = length(β[1])
    return [prod(map(x->x[1]/x[2], zip([shape...]-[idx...]+ones(Int32, dim), shape))) for idx in β]
end

"""
    cumθ{I<:Integer, T<:AbstractFloat}(θ::Array{T,1}, β::Array{Tuple{I,I},1}, shape)

Calculate cumulative sum of θ for each row and column of a matrix.
"""
function cumθ{I<:Integer, T<:AbstractFloat}(θ::Array{T,1}, β::Array{Tuple{I,I},1}, shape)
    ylen, xlen = shape
    cumθ_x = zeros(xlen)
    cumθ_y = zeros(ylen)
    for i in 1:length(β)
        if β[i][1] == 1
            cumθ_x[β[i][2]] = θ[i]
        else
            cumθ_y[β[i][1]] = θ[i]
        end
    end
    cumθ_x = cumsum(cumθ_x)
    cumθ_y = cumsum(cumθ_y)
    return cumθ_x, cumθ_y
end

function genSubβ{T<:AbstractFloat}(β, η::Array{T,2}; ϵ=1e-6)
    subβ_ = sort(β, 1; by=pos->η[pos...], rev=true)
    #subβ = [subβ_[1]]
    subβ = [1]
    for i = 2:length(subβ_)
        if abs(η[subβ[end]...] - η[subβ_[i]...]) > ϵ
            #append!(subβ, [subβ_[i]])
            push!(subβ, i)
        else
            common = tuple(map(x->max(x...), zip(subβ[end], subβ_[i]))...)
            if abs(η[subβ[end]...] - η[common...]) > ϵ
                #append!(subβ, [subβ_[i]])
                push!(subβ, i)
            end
        end
    end
    return subβ
end

"""
        nBalancing{T<:AbstractFloat}(P::Array{T,2}; ϵ=1e-9, max_iter=100, step=1.0, log_interval=10)

Execute balancing by e-projection.
"""
function nBalancing{T<:AbstractFloat}(P::Array{T,2}; ϵ=1e-9, max_iter=100, step=1.0, log_interval=10)
    A = copy(P)
    ylen, xlen = size(A)
    β = genβ(A)
    η_target = genTargetη(A, β)

    # allocate memory
    θ, λ = zeros(length(β)), 1.0
    ∇F = zeros(length(β) + 1)
    jacobian = zeros(length(β) + 1, length(β) + 1)

    F = function(θ, λ)
        sm = sum(A)
        log(sm) - (η_target'θ)[1] + λ * (sm - 1)
    end

    # e-projection loop
    for count = 1:max_iter
        sm = sum(A)
        η = calcη(A) ./ sm
        for i in 1:length(β)
            ∇F[i] = (1 + λ * sm) * η[β[i]...] - η_target[i] 
        end
        ∇F[end] = sm - 1

        l2norm = sqrt(sum(∇F.^2))
        if mod(count, log_interval) == 0
            debug("[step $count] obj.: $(F(θ, λ)), grad: $l2norm, lambda: $λ")
        end
        if l2norm < ϵ break end

        # update jacobian
        for i in 1:length(β)
            for j in i:length(β)
                common = tuple(map(x->max(x...), zip(β[i], β[j]))...)
                jacobian[i,j] = (1 + λ * sm) * η[common...] - η[β[i]...] * η[β[j]...]
                jacobian[j,i] = jacobian[i,j]
            end
            jacobian[i,end] = jacobian[end,i] = sm * η[β[i]...]
        end

        # update parameters
        δθλ = zeros(length(β) + 1)
        try
            δθλ = step * (jacobian \ (-∇F)) # solve a linear equation
        catch
            warn("Jacobian matrix is singular. Projection loop is terminated.")
            break
        end
        θ += δθλ[1:end-1]
        λ += δθλ[end]
        # make cumulative sum of θ for fast update
        cumθ_x, cumθ_y = cumθ(θ, β, size(A))
        # update matrix
        for j = 1:xlen, i = 1:ylen
            if P[i,j] > 1e-15
                A[i,j] = P[i,j] * exp(cumθ_y[i] + cumθ_x[j])
            end
        end
    end

    cumθ_x, cumθ_y = cumθ(θ, β, size(A))
    r = exp.(cumθ_y)
    s = exp.(cumθ_x)
    return A, r, s
end

include("FO.jl")

end # module
