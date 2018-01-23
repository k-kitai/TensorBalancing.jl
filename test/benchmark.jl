using BenchmarkTools
using TensorBalancing
using Logging
include("knight_ruiz.jl")

Logging.configure(level=ERROR)

TB = TensorBalancing
if TB.USE_AF
    using ArrayFire
end

include("common.jl")

Ns = round.(Int32, exp.(linspace(log(100), log(30000), 15)))
Ns = Ns[1:11]

TB.nBalancing(Hessenberg_mod(3))
TB.qnBalancing(Hessenberg_mod(3))
TB.qnBalancing_double(Hessenberg_mod(3))
TB.skBalancing(Hessenberg_mod(3))
knight_ruiz(Hessenberg_mod(3))

if TB.USE_AF
    TB.qnBalancing(AFArray(Hessenberg_mod(3)))
else
    AFArray(x) = Void
end

macro average_n_times(n, expr)
    quote
        t = 0.0
        for i = 1:$n
            t += @elapsed $(esc(expr))
        end
        t / $n
    end
end

TESTS_TO_DO = 1:3
if !isempty(ARGS)
    TESTS_TO_DO = parse.(Int, ARGS)
end

if 1 in TESTS_TO_DO && TB.USE_AF
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
        sp = squeeze(sp)
        sq = Array(squeeze(sp, 1))
        afsp = AFArray(sq)

        time1 = @average_n_times 5 TB.qnBalancing(sq, 1.0e-6, 2^30)
        @printf "%12.5f\t" time1
        time2 = @average_n_times 5 TB.qnBalancing_double(sq, 1.0e-6, 2^30)
        @printf "%12.5f\t" time2
        time3 = @average_n_times 5 knight_ruiz(sq, 1.0e-6)
        @printf "%12.5f\t" time3
        time4 = @average_n_times 5 TB.skBalancing(sq, 1.0e-6, NaN)
        @printf "%12.5f\t" time4
        time5 = @average_n_times 5 TB.nBalancing(sq, 1.0e-6, NaN);
        @printf "%12.5f\n" time5
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

if 5 in TESTS_TO_DO
    println("=== profiling ===")
    sp, m = load_csc_file("Hi-C_example/x_res50000.txt")
    sq = squeeze(sp, 1)
    @profile TB.qnBalancing(sq, 1.0e-6, 2^30)
    Profile.print(mincount=300)
end

if 6 in TESTS_TO_DO
    println("=== logging ϵ ===")
    default_stdout = STDOUT

    sp, m = load_csc_file("Hi-C_example/x_res5000.txt")
    sq = Array(squeeze(sp, 1))

    write(default_stdout, "Executing qnBalancing\n")
    f = open("log_res5000_qnBalancing.txt", "w")
    redirect_stdout(f)
    X = TB.qnBalancing(sq, 1.0e-6, log_norm=true);
    @assert TB.calcRes(X) < 1.0e-6
    close(f)

    write(default_stdout, "Executing qnBalancing_double\n")
    f = open("log_res5000_qnBalancing_double.txt", "w")
    redirect_stdout(f)
    X = TB.qnBalancing_double(sq, 1.0e-6, log_norm=true);
    @assert TB.calcRes(X) < 1.0e-6
    close(f)

    write(default_stdout, "Executing knight_ruiz\n")
    f = open("log_res5000_KnightRuiz.txt", "w")
    redirect_stdout(f)
    X = knight_ruiz(sq, 1.0e-6, log_norm=true);
    @assert TB.calcRes(X) < 1.0e-6
    close(f)

    redirect_stdout(default_stdout)
end
