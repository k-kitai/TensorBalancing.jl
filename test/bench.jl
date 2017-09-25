using TensorBalancing
TB = TensorBalancing

include("SH.jl")
include("FO.jl")
include("metric.jl")

Ns = [5, 10, 22, 50, 100, 224, 500, 1000, 2236, 5000]
#balancing_dict = Dict("n"=>nBalancing, "qn"=>qnBalancing, "sh"=>shBalancing, "kr"=>krBalancing, "ad"=>adamBalancing)
balancing_dict = Dict("n"=>nBalancing, "qn"=>qnBalancing, "sh"=>shBalancing, "ad"=>adamBalancing, "ltqn"=>ltqnBalancing)

#targets = ["n", "qn", "sh", "kr"]
targets = ["n", "qn", "sh"]
if length(Base.ARGS) > 0
    targets = Base.ARGS
end

#accuracies = [3,4,5]
accuracies = [3]
mkpath("./bench")
for accuracy = accuracies
    ϵ = 0.1^accuracy
    for target = targets
        print("Benchmarking $(target)Balancing()\n")
        f = open("./bench/$(target)_$(accuracy)_log.txt", "w")
        write(f, "# N, time\n")
  
        # instantiate function
        A = genHessenberg(100)
        balancing = balancing_dict[target]
        balancing(A)
    
        times = ones(length(Ns))*NaN
        for i = 1:length(Ns)
            A = genHessenberg(Ns[i])
            B = zeros(size(A))
            if i > 1 && (isnan(times[i-1]) || times[i] > 500)
                times[i] = NaN
                continue
            end
            tm = 0.0
            x = zeros(Ns[i])
            for _ = 1:3
                tm += @elapsed begin
                    if target == "qn" || target == "ltqn"
                        B = balancing(A; ϵ=ϵ, max_iter=5, show_trace=true, show_every=1)
                        #B = balancing(A; ϵ=ϵ, max_iter=10000, limited=false)
                    #elseif target == "kr"
                    #    x, _ = balancing(vcat(hcat(zeros(size(A)[1], size(A)[1]), A), hcat(transpose(A), zeros(size(A)[2], size(A)[2]))), ϵ)
                    else
                        #B = balancing(A; ϵ=ϵ, max_iter=10000, show_trace=true)
                        B = balancing(A; ϵ=ϵ, max_iter=10000000)
                    end
                end
                if isnan(calcError(B))
                    break
                end
            end
            #if target == "kr"
            #    B = diagm(x) * vcat(hcat(zeros(size(A)[1], size(A)[1]), A), hcat(transpose(A), zeros(size(A)[2], size(A)[2]))) * diagm(x)
            #    B = B[1:Ns[i], Ns[i]+1:end]
            #end
  
            l2error = 0.0
            if target == "sh"
                l2error = calcError(B)
            #elseif target == "kr"
            #    l2error = calcError(B)
            else
                l2error = TB.calcResidual(B.*size(A)[1])
            end
            if l2error > ϵ || isnan(l2error)
                print("l2error = $l2error\n")
                flush(STDOUT)
                tm = NaN
            end
            times[i] = tm / 3.0
            @printf "[acc = %d] N = %4d, %.5f sec elapsed (avg. over 3 times)\n" accuracy Ns[i] times[i]
            flush(STDOUT)
            @printf f "%4d, %.5f\n" Ns[i] times[i]
            flush(f)
        end
        close(f)
    end
end
