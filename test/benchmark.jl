using BenchmarkTools
using ArrayFire
using TensorBalancing

TB = TensorBalancing

Hessenberg(N) = cumsum(eye(N,N), 2)

print("N\tCPU\tGPU\n")
for N = round.(Int32, exp.(linspace(log(100), log(8000), 10)))
    @printf "%4d\t" N
    time1 = @belapsed TB.nBalancing($(Hessenberg(N)), 1.0e-6)
    print(time1, "\t")
    time2 = @belapsed TB.nBalancing_gpu($(Hessenberg(N)), 1.0e-6)
    print(time2, "\n")
end
