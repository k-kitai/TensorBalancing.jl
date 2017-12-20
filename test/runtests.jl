using TensorBalancing
TB = TensorBalancing
using Base.Test

A = [
     1.0 0.5 0.0;
     0.0 0.5 0.5;
     0.0 0.0 0.5
    ]
@test TB.calcRes(A) == sqrt(0.5)
@test TB.applyΔθ(A, [0.0, 0.0, 0.0, log(2)]) == [
                                                 2.0 1.0 0.0;
                                                 0.0 1.0 0.5;
                                                 0.0 0.0 0.5
                                                ]

# Hessenberg matrix
Hessenberg(N) = cumsum(eye(N,N), 2)
A = [
     1.0 1.0 1.0 1.0 1.0;
     0.0 1.0 1.0 1.0 1.0;
     0.0 0.0 1.0 1.0 1.0;
     0.0 0.0 0.0 1.0 1.0;
     0.0 0.0 0.0 0.0 1.0
    ]
A = Hessenberg(5)
@test TB.calcRes(TB.nBalancing(A, 1.0e-9)) < 1e-9
# @test TB.calcRes(TB.nBalancing_gpu(A, 1.0e-9)) < 1e-9
@test norm(TB.nBalancing(Hessenberg(10), 1.0e-9) - eye(10)) < 1e-9
# Random matrix
@test TB.calcRes(TB.nBalancing(rand(5, 5), 1.0e-9)) < 1e-9
# @test TB.calcRes(TB.nBalancing_gpu(rand(5, 5), 1.0e-9)) < 1e-9
