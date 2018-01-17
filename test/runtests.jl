using TensorBalancing
using Logging
TB = TensorBalancing
using Base.Test

# Logging.configure(level=INFO)

A = [
     1.0 0.5 0.0;
     0.0 0.5 0.5;
     0.0 0.0 0.5
    ]
# @test TB.calcRes(A) == sqrt(0.5)
@test TB.applyΔθ(A, [0.0, 0.0, 0.0, log(2)]) == [
                                                 2.0 1.0 0.0;
                                                 0.0 1.0 0.5;
                                                 0.0 0.0 0.5
                                                ]

# Hessenberg matrix
Hessenberg(N) = cumsum(eye(N,N), 2)
function Hessenberg_mod(N)
    H = Hessenberg(N)
    for i = 1:N-1
        H[i+1, i] = 1.0
    end
    H
end

@test TB.calcRes(TB.nBalancing(Hessenberg(10), 1.0e-9)) < 1e-9
@test TB.calcRes(TB.nBalancing(Hessenberg_mod(10), 1.0e-9)) < 1e-9
# @test TB.calcRes(TB.nBalancing_gpu(Float32.(A), 1.0e-4)) < 1e-4
@test norm(TB.nBalancing(Hessenberg(10), 1.0e-9) - eye(10)) < 1e-9
# Random matrix
@test TB.calcRes(TB.nBalancing(rand(5, 5), 1.0e-9)) < 1e-9
# @test TB.calcRes(TB.nBalancing_gpu(rand(Float32, 5, 5), 1.0e-4)) < 1e-4
@test TB.calcRes(TB.recBalancing(Hessenberg_mod(100), 1.0e-7)) < 1e-7
@test TB.calcRes(TB.qnBalancing(Hessenberg_mod(100), 1.0e-7)) < 1e-7
