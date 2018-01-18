using BenchmarkTools
using ArrayFire
using TensorBalancing
using Logging
include("knight_ruiz.jl")

Logging.configure(level=ERROR)

TB = TensorBalancing

include("common.jl")

Ns = round.(Int32, exp.(linspace(log(100), log(30000), 15)))
Ns = Ns[1:11]

TESTS_TO_DO = 1:3
if !isempty(ARGS)
    TESTS_TO_DO = parse.(Int, ARGS)
end

if 1 in TESTS_TO_DO
    print("=== Newton Balancing ===\n")

    print("N\tCPU\tGPU\n")
    for N = Ns
        @printf "%4d\t" N
        time1 = @belapsed TB.nBalancing($(Hessenberg(N)), 1.0e-6)
        print(time1, "\t")
        time2 = @belapsed TB.nBalancing_gpu($(Hessenberg(N)), 1.0e-6)
        print(time2, "\n")
    end
end

if 2 in TESTS_TO_DO
    print("=== Newton vs. Recursive ===\n")

    print("N\tNewton\tRecursive\n")
    for N = Ns
        @printf "%4d\t" N
        time1 = @belapsed TB.nBalancing($(Hessenberg_mod(N)), 1.0e-6)
        print(time1, "\t")
        time2 = @belapsed TB.recBalancing($(Hessenberg_mod(N)), 1.0e-6, NaN, 3)
        print(time2, "\n")
    end
end

include("Hi-C_example/utils.jl")
if 3 in TESTS_TO_DO
    println("=== Capacity-Func vs. Barrier-Func vs. Knight-Ruiz vs. Sinkhorn-Knopp vs. Newton ===\n")

    fnames = ["Hi-C_example/x_res50000.txt",
              "Hi-C_example/x_res25000.txt",
              "Hi-C_example/x_res10000.txt",
              "Hi-C_example/x_res5000.txt"]
    cut_sk = false

    print("filename\tCapacity-Func\tBarrier-Func\tKnight-Ruiz\tSinkhorn-Knopp\tNewton\n")
    for fname = fnames
        print("$(basename(fname))\t")
        sp, m = load_csc_file(fname)
        sq = Array(squeeze(sp, 1))

        # time1 = @elapsed TB.qnBalancing(sq, 1.0e-6, 2^30)
        time1 = @belapsed TB.qnBalancing($(sq), 1.0e-6, 2^30)
        print(time1, "\t")
        # time2 = @elapsed TB.qnBalancing_double(sq, 1.0e-6, 2^30)
        time2 = @belapsed TB.qnBalancing_double($(sq), 1.0e-6, 2^30)
        print(time2, "\t")
        # time3 = @elapsed knight_ruiz(sq, 1.0e-6)
        time3 = @belapsed knight_ruiz($(sq), $(1.0e-6))
        print(time3, "\t")
        # time4 = @elapsed TB.skBalancing(sq, 1.0e-6, NaN)
        time4 = @belapsed TB.skBalancing($(sq), 1.0e-6, NaN)
        print(time4, "\t")
        # time5 = @elapsed TB.nBalancing(sq, 1.0e-6, NaN);
        time5 = NaN # @belapsed TB.nBalancing($(sq), 1.0e-6, NaN);
        print(time5, "\n")
    end
end

if 4 in TESTS_TO_DO
    println("=== capacity vs. barrier vs. Newton-method vs. Knight-Ruiz vs. Sinkhorn-Knopp ===\n")

    cut_sk = false

    print("capacity\tbarrier\tNewton-method\tKnight-Ruiz\tSinkhorn-Knopp\n")
    for N = Ns
        @printf "%4d\t" N
        time1 = @elapsed TB.qnBalancing(Hessenberg_mod(N), 1.0e-6, 2^30)
        print(time1, "\t")
        time2 = @elapsed TB.qnBalancing_double(Hessenberg_mod(N), 1.0e-6, 2^30)
        print(time2, "\t")
        time3 = @elapsed TB.nBalancing(Hessenberg_mod(N), 1.0e-6, NaN)
        print(time3, "\t")
        time4 = NaN # @elapsed knight_ruiz(symmetrize(Hessenberg_mod(N)), 1.0e-6/sqrt(2))
        print(time4, "\t")
        time5 = NaN
        if !cut_sk
            time5 = @elapsed TB.skBalancing(Hessenberg_mod(N), 1.0e-6, NaN)
        end
        print(time4, "\n")
        if time5 > 10
            cut_sk = true
        end
    end
end

