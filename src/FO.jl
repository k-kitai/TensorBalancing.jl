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

