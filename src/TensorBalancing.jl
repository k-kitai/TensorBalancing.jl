module TensorBalancing
# using ArrayFire

include("matrix/metric.jl")
include("matrix/coordinate.jl")
include("matrix/newton.jl")

######## ALL CONTENTS BELOW ARE DEPRECATED ########

export genβ, calcη, genTargetη, eProject

"""
    genβ{T<:AbstractArray}(P::Array{T,2})

Collect indices for β.
"""
function genβ{T<:AbstractFloat}(P::Array{T,2})
  shape = size(P)
  β = [(1,1)]
  for i in 2:shape[1]
    push!(β, (i,1))
  end
  for i in 2:shape[2]
    push!(β, (1,i))
  end
  return β
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
function genTargetη{T<:AbstractFloat}(P::Array{T}, β)
  _shape = size(P)
  dim = length(β[1])
  return [prod(map(x->x[1]/x[2], zip([_shape...]-[idx...]+ones(Int32, dim), _shape))) for idx in β]
end

"""
    renormalize!(A::Array{T})

Renormalize an array as its elements sums to 1.

# Examples
```julia
julia> A = ones(2,5);
julia> renormalize!(A)
julia> A
2×5 Array{Float64,2}:
 0.1  0.1  0.1  0.1  0.1
 0.1  0.1  0.1  0.1  0.1
```
"""
function renormalize!{T<:AbstractFloat}(P::Array{T})
  sum = 0
  for i in 1:length(P)
    sum += P[i]
  end
  if sum > 0
    for i in 1:length(P)
      P[i] /= sum
    end
  end
  return -log(sum)
end

"""
    eProject{T<:AbstractFloat}(P::Array{T,2}, β, η_target, r=1e-9, max_iter=100)

Execute balancing by e-projection.
"""
function eProject{T<:AbstractFloat}(P::Array{T,2}, β, η_target, r=1e-9, max_iter=100)
  P_size = length(P)
  shape = size(P)
  _P = copy(P)
  # memory allocation
  Δθ = zeros(length(β))
  Δθ[1] = renormalize!(_P)
  δη = zeros(length(β))
  jacobian = zeros(length(β), length(β))

  count = 0
  while true
    η = calcη(_P)
    for i in 1:length(β)
      δη[i] = η_target[i] - η[β[i]...]
    end
    print("L2 error: ", sqrt(sum(δη.^2)), "\n")
    if (sum(δη.^2) < r^2) | (count >= max_iter)
      break
    end
    for i in 1:length(β)
      for j in i:length(β)
        common = tuple(map(x->max(x...), zip(β[i], β[j]))...)
        jacobian[i,j] = η[common...] - η[β[i]...] * η[β[j]...] * P_size
        jacobian[j,i] = jacobian[i,j]
      end
    end
    @show jacobian
    break
    δθ = jacobian \ δη # solve a linear equation
    δθdict = Dict(zip(β, δθ))
    for i in 1:shape[1]
      for j in 1:shape[2]
        factor = exp(sum(map(pos -> δθdict[pos], filter(pos -> (i>=pos[1]) & (j>=pos[2]), β))))
        _P[i,j] = _P[i,j] * factor
      end
    end
    Δθ += δθ
    Δθ[1] += renormalize!(_P)
    count += 1
  end
  return _P, Δθ
end

end # module
