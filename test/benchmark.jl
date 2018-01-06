using BenchmarkTools
using ArrayFire
using TensorBalancing
using Logging

Logging.configure(level=INFO)

TB = TensorBalancing

include("common.jl")

Ns = round.(Int32, exp.(linspace(log(100), log(30000), 15)))
Ns = Ns[1:11]

print("=== Newton Balancing ===\n")

print("N\tCPU\tGPU\n")
for N = Ns
    @printf "%4d\t" N
    time1 = @belapsed TB.nBalancing($(Hessenberg(N)), 1.0e-6)
    print(time1, "\t")
    time2 = @belapsed TB.nBalancing_gpu($(Hessenberg(N)), 1.0e-6)
    print(time2, "\n")
end

print("=== Newton vs. Recursive ===\n")

print("N\tNewton\tRecursive\n")
for N = Ns
    @printf "%4d\t" N
    time1 = @belapsed TB.nBalancing($(Hessenberg_mod(N)), 1.0e-6)
    print(time1, "\t")
    time2 = @belapsed TB.recBalancing($(Hessenberg_mod(N)), 1.0e-6, NaN, 3)
    print(time2, "\n")
end
