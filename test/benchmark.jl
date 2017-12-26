using BenchmarkTools
using ArrayFire
using TensorBalancing
using Logging

Logging.configure(level=INFO)

TB = TensorBalancing

Hessenberg(N) = cumsum(eye(N,N), 2)
function Hessenberg_mod(N)
    H = Hessenberg(N)
    for i = 1:N-1
        H[i+1, i] = 1.0
    end
    H
end

Ns = round.(Int32, exp.(linspace(log(100), log(30000), 15)))

print("=== Newton Balancing ===\n")

print("N\tCPU\tGPU\n")
for N = Ns
    @printf "%4d\t" N
    time1 = @belapsed TB.nBalancing($(Hessenberg(N)), 1.0e-6)
    print(time1, "\t")
    time2 = @belapsed TB.nBalancing_gpu($(Hessenberg(N)), 1.0e-6)
    print(time2, "\n")
end
