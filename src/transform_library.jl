using LaTeXStrings, Statistics

"An algebraic sigmoid function."
function sigmoid(x, θ=0.1)
    return x ./ (θ .+ abs.(x))
end

function generate_library(; pairs=True, powers=[], sigmoids=[], custom::Vector{Function})
    # TODO: implement
    error("Not Implemented.")
    transforms = []
    if pairs == True
    end

    for k in powers
    end

    for α in sigmoids
    end

    for fun in custom
        transforms.append(fun)
    end

    return transforms
end

"""
function stat_combine(base_stats)
Combines the columns of the given base_stats SampleStatistics with derived statistics, such as the square root and squared values of each column, and pairs of columns multiplied together.
Returns a new SampleStatistics object containing the derived statistics in addition to the original data.

Inputs
    base_stats: a SampleStatistics object containing the original data and labels.
Outputs
    A SampleStatistics object containing the original data and labels, as well as the derived statistics.
Example
    data = [1 2; 3 4]
    labels = ["a", "b"]
    base_stats = SampleStatistics(data, labels)
    derived_stats = stat_combine(base_stats)
    # derived_stats will contain the original data and labels from base_stats, as well as the square root, squared, and multiplied values of each column.
"""
function stat_combine(base_stats)  #, transforms::Vector{Function}=nothing)
    m, n = size(base_stats.data)
    n_pairs = Int(n * (n - 1) / 2)
    super_data = Matrix{Float64}(undef, m, 5n + 1n_pairs)
    super_data[:, 1:n] = base_stats.data
    super_stats = SampleStatistics(super_data, base_stats.labels)
    
    # simple functions
    k = 1
    for (data, label) in zip(eachcol(base_stats.data), base_stats.labels)
        # sign = data ./ max.(abs.(data), eps)
        super_stats.data[:, n+k] = abs.(data) .^ (0.5)
        push!(super_stats.labels, L"\|" * label * L"\|^{.5}")
        k += 1
        super_stats.data[:, n+k] = data .^ 2
        push!(super_stats.labels, L"(" * label * L")^{2}")
        k += 1
        super_stats.data[:, n+k] = sigmoid(data, 0.5)
        push!(super_stats.labels, L"S_{0.5}(" * label * L")")
        k += 1
        super_stats.data[:, n+k] = sigmoid(data, 0.1)
        push!(super_stats.labels, L"S_{0.1}(" * label * L")")
        k += 1
        # super_stats.data[:, n+k] = 1.0 ./ data
        # push!(super_stats.labels, L"( 1/" * label * L")")
        # k += 1
    end
    
    # combine pairs products and divisors
    for (i, (data_i, label_i)) in enumerate(zip(eachcol(base_stats.data), base_stats.labels))
        for (j, (data_j, label_j)) in enumerate(zip(eachcol(base_stats.data), base_stats.labels))
            if j >= i
                continue
            end
            super_stats.data[:, n+k] = data_i .* data_j
            push!(super_stats.labels, label_i * label_j)
            k += 1
            # super_stats.data[:, n+k] = data_i ./ data_j
            # push!(super_stats.labels, L"(" * label_i * L"/"* label_j * L")")
            # k += 1
            # super_stats.data[:, n+k] = data_j ./ data_i
            # push!(super_stats.labels, L"(" * label_j * L"/"* label_i * L")")
            # k += 1
            # super_stats.data[:, n+k] = 1.0 ./ data_j ./ data_i
            # push!(super_stats.labels, L"(1/" * label_j * label_i * L")")
            # k += 1
        end
    end

    # degree 2 polynomial kernel
    # for (i, (data_i, label_i)) in enumerate(zip(eachcol(base_stats.data), base_stats.labels))
    #     for (j, (data_j, label_j)) in enumerate(zip(eachcol(base_stats.data), base_stats.labels))
    #         if j >= i
    #             continue
    #         end
    #         super_stats.data[:, 1n+k] = (data_i .* data_j .+ 1.0).^2
    #         push!(super_stats.labels, L"((" * label_i * label_j * "+ 1)^2)")
    #         k += 1
    #     end
    # end

    # radial basis kernel function
    # σ = 1
    # for (i, (data_i, label_i)) in enumerate(zip(eachcol(base_stats.data), base_stats.labels))
    #     for (j, (data_j, label_j)) in enumerate(zip(eachcol(base_stats.data), base_stats.labels))
    #         if j >= i
    #             continue
    #         end
    #         super_stats.data[:, 1n+k] = exp.(-sum((data_i .- data_j).^2.0, dims=2) ./ 2.0 ./ σ^2)
    #         push!(super_stats.labels, L"(exp(-|" * label_i * " - " * label_j * "|^2/2)")
    #         k += 1
    #     end
    # end

    # degree 2 polynomial on inverse of variables
    # for (i, (data_i, label_i)) in enumerate(zip(eachcol(base_stats.data), base_stats.labels))
    #     for (j, (data_j, label_j)) in enumerate(zip(eachcol(base_stats.data), base_stats.labels))
    #         if j >= i
    #             continue
    #         end
    #         super_stats.data[:, 1n+k] = (data_i ./ data_j .+ 1.0).^2
    #         push!(super_stats.labels, L"((" * label_i * L"/" * label_j * "+ 1)^2)")
    #         k += 1
    #     end
    # end
    # for (i, (data_i, label_i)) in enumerate(zip(eachcol(base_stats.data), base_stats.labels))
    #     for (j, (data_j, label_j)) in enumerate(zip(eachcol(base_stats.data), base_stats.labels))
    #         if j >= i
    #             continue
    #         end
    #         super_stats.data[:, 1n+k] = (data_j ./ data_i .+ 1.0).^2
    #         push!(super_stats.labels, L"((" * label_j * L"/" * label_i * "+ 1)^2)")
    #         k += 1
    #     end
    # end
    # for (i, (data_i, label_i)) in enumerate(zip(eachcol(base_stats.data), base_stats.labels))
    #     for (j, (data_j, label_j)) in enumerate(zip(eachcol(base_stats.data), base_stats.labels))
    #         if j >= i
    #             continue
    #         end
    #         super_stats.data[:, 1n+k] = (1.0 ./ data_i ./ data_j .+ 1.0).^2
    #         push!(super_stats.labels, L"((" * L"1/" * label_i * L"/" * label_j * "+ 1)^2)")
    #         k += 1
    #     end
    # end
    @assert size(super_stats.data)[2] == length(super_stats.labels)
    return super_stats
end

"Normalise each column of data to have unit standard deviation and zero mean"
function normalise_sigma(X::SampleStatistics)
    σ_X = zeros(length(X.labels))
    μ_X = zeros(length(X.labels))
    for (i, stat) in enumerate(eachcol(X.data))
        σ_X[i] = std(stat)
        μ_X[i] = mean(stat)
        X.data[:, i] = (stat .- μ_X[i]) ./ σ_X[i]
    end
    return X, σ_X, μ_X
end


function inv_normalise_sigma(X::Matrix, σ_X, μ_X)
    for (i, stat) in enumerate(eachcol(X))
        X[:, i] = stat .* σ_X[i] .+ μ_X[i]
    end
    return X
end


"Return true only if no NaN or Infinity in vector"
function any_bad(x::Vector)
    s = sum(x)
    return isnan(s) || !isfinite(s)
end

"Normalise each column of data to have unit standard deviation and zero mean"
function normalise_sigma!(X::SampleStatistics)
    σ_X = zeros(length(X.labels))
    μ_X = zeros(length(X.labels))
    for (i, stat) in enumerate(eachcol(X.data))
        σ_X[i] = std(stat)
        μ_X[i] = mean(stat)
        X.data[:, i] = (stat .- μ_X[i]) ./ σ_X[i]
    end
    return nothing
end


"Normalise each column of data to a unit range."
function normalise_range(X::SampleStatistics)
    min_X = zeros(length(X.labels))
    max_X = zeros(length(X.labels))
    for (i, stat) in enumerate(eachcol(X.data))
        min_X[i] = minimum(stat)
        max_X[i] = maximum(stat)
        dx = max_X[i] - min_X[i]
        if dx > eps
            X.data[:, i] = (stat .- min_X[i]) ./ dx
        end
    end
    return X, min_X, max_X
end