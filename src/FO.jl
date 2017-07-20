# FO.jl
# Copyright 2017 Koki Kitai.
# 
# First-order gradient descent methods

function shBalancing{T<:AbstractFloat}(P::Array{T,2}; ϵ=1e-9, max_iter=100, log_interval=10)
    ylen, xlen = size(P)
    P = copy(P)
    β = genβ(P)
    η_target = genTargetη(P, β)
    for count = 1:max_iter
        for j = 1:xlen
            sm = sum(P[:,j])
            P[:,j] ./= sm * xlen
        end
        for i = 1:ylen
            sm = sum(P[i,:])
            P[i,:] ./= sm * ylen
        end
        η = calcη(P)
        l2norm = sqrt(sum([η[β[i]...] - η_target[i] for i in 1:length(β)].^2))
        if mod(count, log_interval) == 0
            debug("[step $count] obj.: l2norm: $l2norm")
        end
        if l2norm < ϵ
            debug("[step $count] obj.: l2norm: $l2norm")
            break
        end
    end
    return P
end

"""
    gdBalancing{T<:AbstractFloat}(P::Array{T,2}; learning_rate=1e-2, ϵ=1e-9, max_iter=100, log_interval=10)

Balancing by Gradient Descent
"""
function gdBalancing{T<:AbstractFloat}(P::Array{T,2}; learning_rate=1e-2, ϵ=1e-9, max_iter=100, log_interval=10)
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
        ∇F *= learning_rate

        l2norm = sqrt(sum(∇F.^2))
        if mod(count, log_interval) == 0
            debug("[step $count] obj.: $(F(θ, λ)), grad: $l2norm, lambda: $λ")
        end
        if l2norm < ϵ break end

        # update parameters
        δθλ = zeros(length(β) + 1)
        while ∇F[end] > 0 && ∇F[end] > λ
            ∇F /= 2
        end
        θ -= ∇F[1:end-1]
        λ -= ∇F[end]
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
