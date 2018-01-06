Hessenberg(N) = cumsum(eye(N,N), 2)
function Hessenberg_mod(N)
    H = Hessenberg(N)
    for i = 1:N-1
        H[i+1, i] = 1.0
    end
    H
end

function symmetrize(X)
    M, N = size(X)
    vcat(hcat(zeros(M,M), X), hcat(X', zeros(N,N)))
end

