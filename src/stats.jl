using Convex, SCS

function lstq(A, b)
    return A \ b
end

function bootstrap(A, b, func::Function, n_resamples=10000, output_size=nothing; func_args...)
    if output_size === nothing
        n, m = size(A)
    else
        n, m = output_size
    end
    x = zeros(m, n_resamples)
    for i in 1:n_resamples
        sampleindex = rand(1:n, n)
        x[:, i] = func(A[sampleindex, :], b[sampleindex]; func_args...)
    end
    return x
end

function lasso_loss(b, p)
    Q, c, γ, λ = p
    L1 = transpose(b) * Q * b
    L2 = transpose(c) * b                 #c'b
    L3 = sum.(abs.(b))              #sum(|b|)
    L4 = sumsquares(b)            #sum(b^2)
    if λ > 0
        return L1 - 2 * L2 + γ * L3 + λ * L4      #u'u/T + γ*sum(|b|) + λ*sum(b^2), where u = Y-Xb
    else
        return L1 - 2 * L2 + γ * L3               #u'u/T + γ*sum(|b|) where u = Y-Xb
    end
end
