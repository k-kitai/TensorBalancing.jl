using Optim
using LineSearches
using Parameters
using NLSolversBase

using TensorBalancing
TB = TensorBalancing

function adamBalancing{T<:AbstractFloat}(P::Array{T,2}; ϵ=1e-6, max_iter=1000, show_trace=false, show_every=100)
    A = copy(P)
    ylen, xlen = size(A)
    cumθ_y, cumθ_x = zeros(ylen), zeros(xlen)
    η = zeros(xlen+ylen-1)
    β = TB.genβ(A::Array{T,2})
    β = β[2:end]
    initial_η = zeros(size(A))
    TB.calcη!(initial_η, A)
    η_target = TB.genTargetη(A, β)

    function fg!(grad, θ)
        TB.cumθ!(cumθ_x, cumθ_y, θ, β, size(A))
        local s = 0.0
        for j = 1:xlen, i = 1:ylen
            if P[i,j] > 1e-15
                A[i,j] = P[i,j] * exp(cumθ_y[i] + cumθ_x[j])
                s += A[i,j]
            end
        end

        TB.calcMarginals!(η, A)
        η ./= η[1]
        for i in 1:length(β)
            grad[i] = η[i+1] - η_target[i]
        end

        return log(s) - η_target'θ
    end

    colsums = sum(A, 1)'
    colscalelog = -log.(colsums)

    # hyper parameters
    α = 0.5
    β1 = .9
    β2 = .999
    ϵ_adam = 1e-8
    count = 0

    # state of bfgs
    x        = zeros(length(β))
    g        = zeros(length(β))
    val      = 0.0
    val_prev = 0.0
    m        = zeros(length(β)) # 1st moment
    v        = zeros(length(β)) # 2nd moment

    val = fg!(g, x)
    res = TB.calcResidual(A, xlen/sum(A))

    while res >= ϵ && count < max_iter && val != Inf
        #if show_trace && count % show_every == 0
            @printf "res: %.10f\tval: %.10f\n" res val
        #end
        val_prev = val

        m .= β1*m + (1-β1)g
        v .= β2*v + (1-β2)*(g.^2)

        α = 1.0
        val = fg!(g, x.-α*m./(sqrt.(v)+ϵ_adam))
        new_res = TB.calcResidual(A, xlen/sum(A))
        incount = 1
        while (val - val_prev) > 0 #|| d'g < σ1*d'*g_prev
            α *= .9
            val = fg!(g,  x.-α*m./(sqrt.(v)+ϵ_adam))
            new_res = TB.calcResidual(A, xlen/sum(A))
            if incount == 50 break end
            incount += 1
        end
        x .-= α*m./(sqrt.(v)+ϵ_adam)
        val = fg!(g, x)

        res = TB.calcResidual(A, xlen/sum(A))
        count += 1
    end
    show_trace && @printf "res: %.10f\tval: %.10f\n" res val

    return A./sum(A)
end
