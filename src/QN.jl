# QN.jl
# Copyright 2017 Koki Kitai.
# 
# Quasi-Newton methods

"""
    qnBalancing{T<:AbstractFloat}(P::Array{T,2}; max_iter=100, L=10, log_interval=10)

E-projection by AdaQN.

# parameters
* P - Input array
* max_iter - Number of iterations
* L - period of updating hessian approximation
* log_interval - interval of reporting optimization
"""
function qnBalancing{T<:AbstractFloat}(P::Array{T,2}; max_iter=100, L=10, log_interval=10, step=1.0)
    ylen, xlen = size(P)
    β = vcat([(1,1)], [(1,i) for i = 2:xlen], [(i,1) for i = 2:ylen])
    η_target = genTargetη(P, β)
    #A = zeros(size(P))
    A = copy(P)
    ψ = function(θ)
        cumθ_x, cumθ_y = cumθ(θ, β, size(A))
        for j = 1:xlen, i = 1:ylen
            #if A[i,j] > 1e-12
                A[i,j] = P[i,j] * exp(cumθ_y[i] + cumθ_x[j])
            #else
            #    A[i,j] = 0
            #end
        end
        scale = renormalize!(A)
        #scale += ret
        return -log(A[1]), scale
    end
    η_β = function(θ)
        ψ(θ)
        η = zeros(xlen + ylen - 1)
        cumcol = zeros(ylen)
        for i = xlen-1:-1:1
            cumcol += A[:,i+1]
            η[i+1] = sum(cumcol)
        end
        cumcol += A[:,1]
        η[xlen+1:end] = reverse(cumsum(reverse(cumcol)))[2:end]
        η[1] = sum(cumcol)
        return η
    end

    f = function(θ)
        y, scale = ψ(θ)
        θ[1] += scale
        return y - (η_target'θ)[1]
    end
    g = θ -> η_β(θ) - η_target
    result = adaqn(f, g, zeros(length(β)), max_iter=max_iter, L=L, log_interval=log_interval, step=step)
    #result[1] += scale
    ψ(result)
    return A
end

"""
    qnBalancing{T<:AbstractFloat}(P::SparseMatrixCSC{T,Int64}; step=1.0, max_iter=100, L=10, log_interval=10)

A variant of qnBalancing for a sparse matrix.
"""
function qnBalancing{T<:AbstractFloat}(P::SparseMatrixCSC{T,Int64}; step=1.0, max_iter=100, L=10, log_interval=10, ϵ=1e-6)
    P, row_idx, col_idx = shrinkZeros(P)
    ylen, xlen = size(P)
    β = vcat([(1,i) for i = 1:xlen], [(i,1) for i = 2:ylen])
    η_target = genTargetη(P, β)
    A = copy(P)
    rows = rowvals(A)
    ψ = function(θ)
        cumθ_x, cumθ_y = cumθ(θ, β, size(A))
        for j = 1:xlen, i in nzrange(P, j)
            A.nzval[i] = P.nzval[i] * exp(cumθ_y[rows[i]] + cumθ_x[j])
        end
        scale = renormalize!(A)
        return -log(A[1]), scale
    end
    η_β = function(θ)
        ψ(θ)
        η = zeros(xlen + ylen - 2)
        cumcol = zeros(ylen)
        for i = xlen-1:-1:1
            cumcol += A[:,i+1]
            η[i] = sum(cumcol)
        end
        cumcol += A[:,1]
        η[xlen:end] = reverse(cumsum(reverse(cumcol)))[2:end]
        return η
    end
    f = function(θ)
        y, scale = ψ(θ)
        #θ[1] += scale
        return y - (η_target'θ)[1] #- η_target[1] * scale
    end
    g = θ -> η_β(θ) - η_target
    result = adaqn(ψ, g, zeros(length(β)); step=step, max_iter=max_iter, L=L, log_interval=log_interval)
    ψ(result)
    return A
end

function adaqnTwoLoopRecursion{T<:AbstractFloat}(S::CircularDeque{Array{T,1}}, Y::CircularDeque{Array{T,1}}, g::Array{T,1}, H0::Array{T,1})
    ρ = [(1 / (x[1]'x[2])[1]) for x in zip(Y, S)]
    τ = length(S)
    q = g
    α = zeros(τ)
    for i = τ:-1:1
        α[i] = ρ[i] * (S[i]'q)[1]
        q -= α[i] .* Y[i]
    end
    r = H0 .* q
    for i = 1:τ
        β = ρ[i] * (Y[i]'r)[1]
        r = r + S[i] * (α[i] - β)
    end
    r
end

function adaqn{T<:AbstractFloat}(f, ∇f, x0::Array{T,1}; step=1.0, L=10, mF=500, mL=100, max_iter=100, log_interval=10, ϵ=1e-6)
    N = length(x0)
    queue_g = CircularDeque{Array{T,1}}(mF)
    queue_S = CircularDeque{Array{T,1}}(mL)
    queue_Y = CircularDeque{Array{T,1}}(mL)

    t = 0
    x_old = zeros(N)
    x_sum = zeros(N)
    x_avg_old = zeros(N)
    score_old = 0
    moment2 = zeros(N) .+ 1e-15
    γ = 1.01

    αs = vcat([[5*.1^n .1^n] for n = 0:8]...)
    x = x0
    for k = 1:max_iter
        g = ∇f(x)
        if mod(k, log_interval) == 0
            debug("[step $k] obj.: $(f(x)), grad: $(sqrt(sum(g.^2)))")
        end
        #print(x, '\n')
        moment2 += g.^2

        H0_diag = 1 ./ sqrt.(moment2)
        #p = adaqnTwoLoopRecursion(queue_S, queue_Y, g, H0_diag)
        #p = H0_diag .* g
        p = g
        α = step
        #n = indmin([f(x - α * p) for α in αs])
        #print("$([f(x - α * p) for α in αs])\n")
        #x = x - αs[n] * p
        x = x - α * p
        x_sum = x_sum + x

        if queue_g.capacity == queue_g.n
            pop!(queue_g)
        end
        push!(queue_g, g)

        if mod(k,L) == 0
            x_avg, x_sum = x_sum/L, 0
            score = f(x_avg)
            if t > 0
                if score > score_old
                    empty!(queue_g)
                    empty!(queue_S)
                    empty!(queue_Y)
                    debug("cleard queues")
                    x = x_avg_old
                    #moment2 = zeros(size(moment2))
                    continue
                end
                s = x_avg - x_avg_old
                y = sum([g*(g's)[1] for g in queue_g]) / length(queue_g)
                if (s'y)[1] > (s's)[1]
                    if queue_S.capacity == queue_S.n
                        pop!(queue_S)
                    end
                    if queue_Y.capacity == queue_Y.n
                        pop!(queue_Y)
                    end
                    push!(queue_S, s)
                    push!(queue_Y, y)
                    x_avg_old = x_avg
                    score_old = score
                end
            else
                x_avg_old = x_avg
                score_old = score
            end
            t = t + 1
        end
    end

    return x_avg_old
end

function calcMarginals(A)
    ylen, xlen = size(A)
    η = zeros(xlen + ylen - 1)
    cumcol = zeros(ylen)
    for i = xlen-1:-1:1
        cumcol += A[:,i+1]
        η[i+1] = sum(cumcol)
    end
    cumcol += A[:,1]
    η[xlen+1:end] = reverse(cumsum(reverse(cumcol)))[2:end]
    η[1] = sum(cumcol)
    return η
end

function qnBalancing2{T<:AbstractFloat}(P::Array{T,2}; ϵ=1e-9, max_iter=100, step=1.0, log_interval=100, L = 10)
    A = copy(P)
    ylen, xlen = size(A)
    β = vcat([(1,1)], [(1,i) for i = 2:xlen], [(i,1) for i = 2:ylen])
    η_target = genTargetη(A, β)

    # allocate memory
    θ, λ = zeros(length(β)), 1.0
    ∇A = zeros(length(β) + 1)

    F = function(θ, λ)
        v = vcat(θ, λ)
        cumθ_x, cumθ_y = cumθ(v, β, size(A))
        for j = 1:xlen, i = 1:ylen
            if P[i,j] > 1e-15
                A[i,j] = P[i,j] * exp(cumθ_y[i] + cumθ_x[j])
            end
        end
        sm = sum(A)
        #log(sm) - (η_target'θ)[1] + λ * (sm - 1)^2
        (1+λ) * log(sm) - (η_target'θ)[1]
    end

    ∇F = function(θ, λ)
        #print(λ, '\n')
        F(θ, λ)
        sm = sum(A)
        η = calcMarginals(A) ./ sm
        for i in 1:length(β)
            #∇A[i] = (1 + λ * sm) * η[i] - η_target[i]
            ∇A[i] = (1 + λ) * η[i] - η_target[i]
        end
        ∇A[end] = -log(sm)
        return ∇A
    end

    f = θλ -> F(θλ[1:end-1], θλ[end])
    g = θλ -> ∇F(θλ[1:end-1], θλ[end])
    print(f(zeros(length(β)+1)), '\n')
    #result = optimize(f, g, zeros(length(β)+1), SimulatedAnnealing())
    adaqn(f, g, vcat(θ, λ), max_iter=max_iter, L=L, log_interval=log_interval, step=step)
    #result[1] += scale
    #ψ(result)
    return A
end
