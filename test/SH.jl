# SH.jl
# Copyright 2017 Koki Kitai.
# 
# Implementation of Sinkhorn balancing.
using TensorBalancing
TB = TensorBalancing

"""
    shBalancing{T<:AbstractFloat}(P::Array{T,2}; ϵ=1e-9)

Sinkhorn balancing on a matrix.
"""
function shBalancing(P; ϵ=1e-9, max_iter=10000, show_trace=false, show_every=100)
    ylen, xlen = size(P)
    P = copy(P)
    β = TB.genβ(P)
    η = zeros(length(β))
    η_target = TB.genTargetη(P, β)

    if show_trace
        @printf "Iter     Residual\n"
        TB.calcMarginals!(η, P)
        #η ./= η[1]
        res = TB.calcResidual(P)
        @printf "%6d   %14e\n" 0 res
    end

    sm = 0.0
    for count = 1:max_iter
        for j = 1:xlen
            sm = sum(P[:,j])
            P[:,j] ./= sm# * xlen
        end
        for i = 1:ylen
            sm = sum(P[i,:])
            P[i,:] ./= sm# * ylen
        end
        TB.calcMarginals!(η, P)
        #η ./= η[1]
        res = TB.calcResidual(P)
        if res < ϵ
            show_trace && @printf "%6d   %14e\n" count res
            break
        end
        if show_trace && count % show_every == 0
            @printf "%6d   %14e\n" count-1 res
        end
    end
    return P
end

"""
    krBalancing{T<:AbstractFloat}(P::Array{T,2}; ϵ=1e-9)

Matrix balancing algorithm of Knight and Ruiz.
"""
function krBalancing{T<:AbstractFloat}(A::Array{T,2}; ϵ=1e-9, max_iter=10000, show_trace=false, show_every=100, delta=0.1, Delta=3, fl=1)

    print(A)

    ylen, xlen = size(A)
    P = zeros(size(A))

    tol = ϵ
    g = 0.9        # Parameters used in inner stopping criterion.
    etamax = 0.1   # Parameters used in inner stopping criterion.
    eta = etamax
    stop_tol = tol * 0.5
    x = ones(xlen)
    rt = tol^2                                 # rt = tol^2
    v = x .* (A * x)                            # v = x.*(A*x)
    rk = 1 - v
    rho_km1 = rk'rk     # rho_km1 = rk'*rk
    rout = rho_km1
    rold = rout

    # x, x0, e, v, rk, y, Z, w, p, ap :     vector shape(n, 1) : [ [value] [value] [value] [value] ... ... ... [value] ]
    # rho_km1, rout, rold, innertol, alpha :  scalar shape(1 ,1) : [[value]]

    MVP = 0                                                      # Well count matrix vector products.
    i = 0                                                        # Outer iteration count.

    if fl == 1
        print("it in. it res\n")
    end

    while rout > rt                                               # Outer iteration
        i = i + 1
        k = 0
        y = e
        innertol = max(eta^2 * rout, rt)                   # innertol = max([eta^2*rout,rt]);

        while rho_km1 > innertol                                # Inner iteration by CG
            k = k + 1
            if k == 1
                Z = rk ./ v                                       # Z = rk./v
                p = Z
                rho_km1 = rk'Z           # rho_km1 = rk'*Z
            else
                beta = rho_km1 / rho_km2
                p = Z + beta * p
            end


            # Update search direction efficiently.
            w = x.*(A*(x.*p))+v.*p
            alpha = rho_km1 / (p'w)
            ap = alpha*p

            # Test distance to boundary of cone.
            ynew = y + ap;
            #print(i, np.amin(ynew), delta, np.amin(ynew) <= delta)
            #print(i, np.amax(ynew), Delta, np.amax(ynew) >= Delta)
            if minimum(ynew) <= delta
                if delta == 0
                    break
                end
                ind = ap .< 0 # ind = find(ap < 0)
                gamma = minimum( (delta - y[ind]) ./ ap[ind] )    # gamma = min((delta - y(ind))./ap(ind))
                y = y + gamma * ap                        # y = y + gamma*ap
                break
            end
            if maximum(ynew) >= Delta
                ind = ynew .> Delta                 # ind = find(ynew > Delta);
                gamma = minimum( (Delta-y[ind]) ./ ap[ind])       # gamma = min((Delta-y(ind))./ap(ind));
                y = y + gamma * ap                        # y = y + gamma*ap;
                break
            end
            y = ynew
            rk = rk - alpha*w                                    # rk = rk - alpha*w
            rho_km2 = rho_km1
            Z = rk ./ v
            rho_km1 = rk'Z               # rho_km1 = rk'*Z
        end

        x = x .* y                                                # x = x.*y
        v = x .* (A*x)                                         # v = x.*(A*x)
        rk = 1 - v
        rho_km1 = rk'rk                  # rho_km1 = rk'*rk
        rout = rho_km1
        MVP = MVP + k + 1

        # Update inner iteration stopping criterion.
        rat = rout/rold
        rold = rout
        res_norm = sqrt(rout)
        eta_o = eta
        eta = g*rat

        #print(i, res_norm)

        if g*eta_o^2 > 0.1
            eta = max(eta, g*eta_o^2)                    # eta = max([eta,g*eta_o^2])
        end

        eta = max(min(eta, etamax), stop_tol/res_norm);   # eta = max([min([eta,etamax]),stop_tol/res_norm]);

        if fl == 1
            #@printf "%3d %6d %.3e %.3e %.3e \n" i k  res_norm minimum(y) minimum(x)
            print("x: $x\n")
            #res=[res, res_norm]
        end

    end

    return P
end
