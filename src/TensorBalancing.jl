module TensorBalancing
export nBalancing, qnBalancing

"""
    genβ{T<:AbstractArray}(P::Array{T,2})

Collect indices for β.
"""
function genβ{T<:AbstractFloat}(P::Array{T,2})
    ylen, xlen = size(P)
    return vcat([(1,1)], [(1,i) for i = 2:xlen], [(i,1) for i = 2:ylen])
end

"""
    calcη!{T<:AbstractArray}(η::Array{T,2}, P::Array{T,2})

Update η coordinate of a matrix.
"""
function calcη!{T<:AbstractFloat}(η::Array{T,2}, P::Array{T,2})
    shape = size(P)
    idx = [shape...]
    η[idx...] = P[idx...]
    for i in shape[1]-1:-1:1
        η[i,shape[2]] = η[i+1, shape[2]] + P[i, shape[2]]
    end
    for j in shape[2]-1:-1:1
        η[shape[1], j] = η[shape[1], j+1] + P[shape[1], j]
    end
    for i in shape[1]-1:-1:1
        for j in shape[2]-1:-1:1
            η[i,j] = η[i+1, j] + η[i, j+1] - η[i+1, j+1] + P[i,j] # de morgan
        end
    end
    η ./= η[1,1]
end

"""
    calcMarginals!{T<:AbstractArray}(η::Array{T,1}, A::Array{T,2})

Update η, a list containing the limb part of cumulative sums of a matrix.
"""
function calcMarginals!{T<:AbstractFloat}(η::Array{T,1}, A::Array{T,2})
    ylen, xlen = size(A)
    cumcol = zeros(ylen)
    for i = xlen-1:-1:1
        cumcol += A[:,i+1]
        η[i+1] = sum(cumcol)
    end
    cumcol += A[:,1]
    for i = length(cumcol)-1:-1:1
      cumcol[i] += cumcol[i+1]
    end
    η[xlen+1:end] = cumcol[2:end]
    η[1] = cumcol[1]
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
    cumθ!(cumθ_x, cumθ_y, θ, β, shape)
    return cumθ_x, cumθ_y
end

"""
    cumθ!{I<:Integer, T<:AbstractFloat}(cumθ_x, cumθ_y, θ::Array{T,1}, β::Array{Tuple{I,I},1}, shape)

Calculate cumulative sum of θ for each row and column of a matrix.
"""
function cumθ!{I<:Integer, T<:AbstractFloat}(cumθ_x::Array{T,1}, cumθ_y::Array{T,1}, θ::Array{T,1}, β::Array{Tuple{I,I},1}, shape)
    ylen, xlen = shape
    for i in 1:length(β)
        if β[i][1] == 1
            cumθ_x[β[i][2]] = θ[i]
        else
            cumθ_y[β[i][1]] = θ[i]
        end
    end
    for i = 2:xlen
        cumθ_x[i] += cumθ_x[i-1]
    end
    for i = 2:ylen
        cumθ_y[i] += cumθ_y[i-1]
    end
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
    nBalancing{T<:AbstractFloat}(P::Array{T,2}; ϵ=1e-9, max_iter=100, step=1.0)

Matrix balancing by E-projection with Newton's method.

# Parameters
* P - Input matrix
"""
function nBalancing{T<:AbstractFloat}(P::Array{T,2}; ϵ=1e-9, max_iter=100)
    A = copy(P)
    ylen, xlen = size(A)
    β = genβ(A)
    η_target = genTargetη(A, β)

    # allocate memory
    η = zeros(size(A))
    θ, λ = zeros(length(β)), 1.0 # with Lagrange multiplier
    δθλ = zeros(length(β)+1)
    grad = zeros(length(β)+1)
    jacobian = zeros(length(β)+1, length(β)+1)
    cumθ_y, cumθ_x = zeros(ylen), zeros(xlen)

    sm = sum(A)
    if sm <= 2.0
        A .*= 2.0 / sm
        θ[1] = log(2.0 / sm)
    end

    # e-projection loop
    for count = 1:max_iter
        sm = sum(A)
        calcη!(η, A)
        for i = 1:length(β)
            grad[i] .= (1 + λ * sm) * η[β[i]...] - η_target[i]
        end
        grad[end] = sm - 1
        if sqrt(sum(grad[1:end-1].^2)) < ϵ
            break
        end

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
        try
            δθλ .= jacobian \ grad # solve a linear equation
        catch
            warn("Jacobian matrix got singular. Projection loop is terminated.")
            break
        end
        θ .-= δθλ[1:end-1]
        λ -= δθλ[end]
        # make cumulative sum of θ for fast update
        cumθ!(cumθ_x, cumθ_y, θ, β, size(A))
        # update matrix
        for j = 1:xlen, i = 1:ylen
            if P[i,j] > 1e-15
                A[i,j] = P[i,j] * exp(cumθ_y[i] + cumθ_x[j])
            end
        end
    end

    return A
end

include("QN.jl")

end # module
