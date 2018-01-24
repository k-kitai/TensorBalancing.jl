Hessenberg(N) = cumsum(eye(N,N), 2)
function Hessenberg_mod(N)
    H = Hessenberg(N)
    for i = 1:N-1
        H[i+1, i] = 1.0
    end
    H
end

function pseudo_hic(N)
    A = zeros(N, N)
    a = [4, 8, 6]
    w = [10, 30, 50]
    for i = 1:N
        width = round(Int64, abs(a' * sin.((π*i)./w)))+1
        for j = i:min(i+width,N)
            A[i,j] = A[j,i] = 1.0 / (j-i+1)
        end
    end
    A
end

function symmetrize(X)
    M, N = size(X)
    vcat(hcat(zeros(M,M), X), hcat(X', zeros(N,N)))
end

