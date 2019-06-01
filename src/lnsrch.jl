"""
Custom backtracking line-search.
Lnsrch returns a step-length α such that the decreasing criterion,
    f(x-αd) < f(x) - γ*|g|
is satisfied.
This is implemented based on other sophisticated line-search methods in LineSearches package.
"""

immutable Lnsrch{TF, TI}
    beta::TF
    gamma::TF
    iterations::TI
    func
end

(ls::Lnsrch)(df, x, s, x_scratch, lsr, alpha, mayterminate) =
    _lnsrch!(df, x, s, x_scratch, lsr, ls.func, alpha, mayterminate,
             ls.beta, ls.gamma, ls.iterations)


function _lnsrch!{T}(df,
                          x::Array{T},
                          s::Array{T},
                          x_scratch::Array{T},
                          lsr::LineSearchResults,
                          lsfunc,
                          alpha::Real = 1.0,
                          mayterminate::Bool = false,
                          beta::Real = 0.8,
                          gamma::Real = 0.01,
                          iterations::Integer = 1_000,
                          maxstep::Real = Inf)
    # Count the total number of iterations
    iteration = 0

    # Count number of parameters
    n = length(x)

    # read f_x and slope from LineSearchResults
    f_x = lsr.value[end]
    gxp = lsr.slope[end]

    # Tentatively move a distance of alpha in the direction of s
    x_scratch .= x .+ alpha.*s

    # Backtrack until we satisfy sufficient decrease condition
    f_x_scratch = lsfunc(x_scratch)
    while isinf(f_x_scratch) || isnan(f_x_scratch)
        alpha *= beta
        x_scratch .= x .+ alpha.*s
        f_x_scratch = lsfunc(x_scratch)
    end
    f_x_scratch = lsfunc(x_scratch)

    while f_x_scratch > f_x + gamma * gxp
        # Increment the number of steps we've had to perform
        iteration += 1

        # Ensure termination
        if iteration > iterations
            break
        end

        alpha *= beta

        # Update proposed position
        x_scratch .= x .+ alpha.*s
        # Evaluate f(x) at proposed position
        f_x_scratch = lsfunc(x_scratch)
        while f_x_scratch == Inf
            alpha *= beta
            x_scratch .= x .+ alpha.*s
            f_x_scratch = lsfunc(x_scratch)
        end
    end

    return alpha
end
