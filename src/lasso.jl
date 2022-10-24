using Convex, SCS
using MLBase
using DataFrames, GLM
using StatsModels: @formula
using LaTeXStrings

include("../src/SampleStatistics.jl")
include("../src/lasso_mlj.jl")


# TODO: use this structure
# struct EN_Model
#     X::Matrix
#     labels::Vector
#     Y::Vector
#     intercept::Bool
#     λ
#     γ
#     β::Vector
#     idx_selected::Vector{Integer}
#     groups
# end

# EN_Model(X, labels, Y, intercept) = EN_Model(X, labels, Y, intercept, λ=Nan, γ=Nan, β=[], idx_selected=[], groups=[])

function add_unit_column(A)
    return [A ones(eltype(A), size(A, 1))]
end

""" Wrapper to LassoENConstrained(A::Matrix, ...)
For consistency with MLJ's signatures giving model as the first argument.
"""
function LassoENConstrained(model)
    model.β = LassoENConstrained(model.X, model.Y; groups=(), γ=0.1, λ=0.1, intercept=false)
    model.selected = findall(x -> abs.(x) < eps, model.β)
    return model
end

"""
    LassoEN(A,b;groups, γ,λ, intercept)

Do Lasso Regresseion (set γ>0,λ=0), ridge regression (set γ=0,λ>0) or elastic net regression (set γ>0,λ>0).


# Input
- `A::VecOrMat`:   TxK matrix of covariates (regressors)
- `b::Vector`:     T-vector with the response (dependent) variable
- 'groups':        Tuple of parameter indexes to constrain by A[groups[:, g]*X[g] = 0] for g ∈ group
- `γ::Number`:     penalty on sum(abs.(b))
- `λ::Number`:     penalty on sum(b.^2)

Blatantly copied from Convex.jl examples, but with constrains.

"""
function LassoENConstrained(X::Matrix, y; groups=(), γ=0.1::Float64, λ=0.1::Float64, intercept=false)
    if intercept
        X = add_unit_column(X)
    end

    n_vars = size(X, 2)
    Q = X'X / size(X, 1)
    c = X'y / size(X, 1)
    β = Variable(n_vars)

    # for param_group in groups
    #     add_constraint!(β, sum(X[:, param_group] * β[param_group]) == 0)
    # end

    if γ > 0 && λ > 0
        problem = minimize(quadform(β, Q) - 2 * dot(c, β) + γ * norm(β, 1) + λ * sumsquares(β))
    elseif γ > 0
        problem = minimize(quadform(β, Q) - 2 * dot(c, β) + γ * norm(β, 1))
    elseif λ > 0
        problem = minimize(quadform(β, Q) - 2 * dot(c, β) + λ * sumsquares(β))
    else
        problem = minimize(quadform(β, Q) - 2 * dot(c, β))
    end

    solve!(problem, SCS.Optimizer, verbose=false, silent_solver=true)
    return vec(β.value)
end

function compute_rmse(c::Vector{Float64}, X::Matrix{Float64}; intercept=false, by_col=false)
    if by_col
        X = transpose(X)
    end

    if intercept
        return sqrt(mean(sum(abs2.(X .- c[1:end-1] .+ c[end]); dims=1)))
    else
        return sqrt(mean(sum(abs2.(X .- c); dims=1)))
    end
end

function compute_rmse(c::Vector{Float64}, X::Vector{Float64}; intercept=false)
    if intercept
        return sqrt(sum(abs2.(X .- c[1:end-1] .+ c[end])))
    else
        return sqrt(sum(abs2.(X .- c)))
    end
end

function compute_model_rmse(β, X, Y, inds)
    Y_predict = X[inds, :] * β[1:end-1] .+ β[end]
    Y_true = Y[inds]
    return compute_rmse(Y_predict, Y_true, intercept=false)
end

function compute_model_rmse(β, X, Y)
    Y_predict = X * β[1:end-1] .+ β[end]
    Y_true = Y
    return compute_rmse(Y_predict, Y_true, intercept=false)
end

"Predicted residual error sum of squares

This function computes the predicted residual error sum of squares (PRESS) for a given set of regression coefficients, predictor matrix, and response variable.
The PRESS statistic is a measure of model fit and is commonly used in cross-validation to assess the performance of a regression model.
It is calculated by looping through each sample, computing the predicted value for the response variable, and summing the squared differences between the observed and predicted values.
The PRESS value is then normalized by the squared values of the diagonal elements of the hat matrix.
A smaller PRESS value indicates a better fit of the model to the data.
"
function compute_press(β, X, Y, eps=1e-8)
    # Compute the adjusted X matrix and its inverse
    X_adjoint = X' * X + eps * I
    X_adjoint_inv = inv(X_adjoint)

    # Compute the Hat matrix and Y_predict
    H = X * X_adjoint_inv * X'
    Y_predict = X * β[1:end-1] .+ β[end]

    # Initialize the press value
    press = 0.0

    # Loop through the samples and compute the press
    n = size(X)[1]
    for i in 1:n
        press += (Y[i] - Y_predict[i])^2 / (1 - H[i, i])^2
    end

    # Return the computed press value
    return press
end

function compute_p_squared(Y, press)
    dely = sum((Y .- mean(Y)) .^ 2)
    return press / dely
end

function compute_press_brute(X, Y)
    @warn "Not tested."
    n = size(X)[1]
    press = 0.0
    for i in 1:n
        if i == 1
            X_not_i = X[2:end, :]
        elseif i == n
            X_not_i = X[1:end-1, :]
        else
            X_not_i = [X[1:i, :] X[i+1, end, :]]
        end
        β_not_i = LassoENConstrained(X, Y; λ=0.1, γ=0.1, intercept=true, groups=[])
        Y_predict_not_i = X_not_i * β_not_i[1:end-1] .+ β_not_i[end]
        press += (Y[i] .- Y_predict_not_i) .^ 2
    end
    return press
end

function train_convex(X::SampleStatistics, Y, groups; intercept=true, alphas=[1.0], l1_ratios=[1.0])
    return train_convex(X.data, X.labels, Y, groups, intercept=intercept, alphas=alphas, l1_ratios=l1_ratios)
end

function train_convex(X::Array, X_labels, Y, groups; intercept=true, alphas=[1.0], l1_ratios=[1.0], cv=false)
    n_vars = size(X)[2]
    if intercept && !(X_labels[end] == L"1")
        push!(X_labels, L"1")
        n_vars += 1
    end

    # train, validate split
    X, X_val, Y, Y_val = train_test_split(X, vec(Y), split=0.8)

    γ = reshape(alphas * l1_ratios', :)
    λ = reshape(alphas * (1.0 .- l1_ratios)', :)

    if cv == false && length(γ) == 1 && length(λ) == 1
        coefs = LassoENConstrained(X, Y; γ=γ[1], λ=λ[1], intercept=intercept, groups=groups)
        loss = compute_model_rmse(coefs, X_val, Y_val)
    else
        error("Implementation error.")
        # grid seach for best hyper-parameters
        X_train, X_val, Y_train, Y_val = train_val_split(X, X_labels, vec(Y); split=0.8)
        estfun = (γ, λ) -> LassoENConstrained(X_train, Y_train; γ=γ, λ=λ, intercept=intercept, groups=groups)
        evalfun = (coefs) -> compute_model_rmse(coefs, X_val, Y_val)
        γ_grid = ("γ", γ)
        λ_grid = ("λ", λ)
        (coefs, _, _) = gridtune(estfun, evalfun, γ_grid, λ_grid, ord=Reverse, verbose=false)

        # unbiased loss value on val data
        loss = compute_model_rmse(best_model, X_val, Y_val)
    end

    return coefs, loss
end

function list_terms(β, labels; importance=0.1, omit_final=true, silent=false)
    # omit final variable (e.g. constant)
    if omit_final
        abs_coefs = abs.(β[1:end-1])
    else
        abs_coefs = abs.(β)
    end
    select = abs_coefs / maximum(abs.(abs_coefs)) .> importance
    if !silent
        println("")
        for i in findall(select)
            println("$(labels[i])\t $(β[i])")
        end
    end
    return select
end

""" Train an Elastic Net Regression 
Do binary search over alpha to target n_terms parameters in the equation.

This function trains a regression model using either the convex optimizer or the JML coordinate descent optimizer.
The number of terms in the model can be specified and the training will continue until that number of terms is reached.
The function also allows for different values of the regularization parameter alpha and the l1 ratio to be tested.
The function returns the optimized coefficients and loss value for the trained model.
"""
function train(X, Y, groups; intercept=true, n_terms_target=1, start_alpha=1e-3, l1_ratios=[1.0], use_convex=true, max_iterations=100, alpha_low=1e-12, alpha_high=1e12, log_search=false, importance=0.001)

    @assert start_alpha <= alpha_high
    @assert start_alpha >= alpha_low

    converged = false
    alpha = start_alpha
    iterations = 0
    β = nothing
    loss = nothing
    n_terms_found::Int = 0


    while !converged && iterations < max_iterations
        if use_convex
            # use convex's optimiser
            β, loss = train_convex(X, Y, groups, intercept=intercept, alphas=alpha, l1_ratios=l1_ratios)
        else
            # use JML's coordinate descent optimiser with no group constraints
            β = train_SKL(X, Y, intercept=intercept, alphas=[alpha], l1_ratios=l1_ratios)
        end
        select = list_terms(β, X.labels; importance=importance, omit_final=use_convex, silent=true)

        # assess number of terms found
        n_terms_found = count(select)
        println("$n_terms_found terms using alpha = $alpha.")
        if n_terms_found == n_terms_target
            converged = true
        elseif n_terms_found > n_terms_target
            alpha_low = alpha
        elseif n_terms_found < n_terms_target
            alpha_high = alpha
        end
        if alpha_high <= alpha_low
            error("Bad alpha range given at $alpha. Try making it wider.")
        end

        # update alpha using binary search in log space
        if log_search
            alpha = exp10((log10(alpha_high) + log10(alpha_low)) / 2.0)
        else
            alpha = alpha_low + (alpha_high - alpha_low) / 2.0
        end
        iterations += 1
    end
    if iterations == max_iterations
        error("Training exceeded maximum iterations.
        Still have $n_terms_found terms involved, instead of $n_terms_target targeted.
        Last value of alpha was $alpha.")
    end

    return β
end

function get_beta_unbiased(y, X, σ_X, μ_X, select)
    # linear fit coefficients on un-normalised data

    n_terms_target = length(select)
    if n_terms_target == 0
        data = DataFrame(y=vec(y))
        ols = lm(@formula(y ~ 1), data)
    elseif n_terms_target == 1
        X_subset1 = X.data[:, select] .* σ_X[select[1]] .- μ_X[select[1]]
        data = DataFrame(y=vec(y), x1=vec(X_subset1))
        ols = lm(@formula(y ~ 1 + x1), data)
    elseif n_terms_target == 2
        i = select[1]
        j = select[2]
        X_subset1 = X.data[:, i] .* σ_X[i] .- μ_X[i]
        X_subset2 = X.data[:, j] .* σ_X[j] .- μ_X[j]
        data = DataFrame(y=vec(y), x1=vec(X_subset1), x2=vec(X_subset2))
        ols = lm(@formula(y ~ 1 + x1 + x2), data)
    elseif n_terms_target == 3
        i = select[1]
        j = select[2]
        k = select[3]
        X_subset1 = X.data[:, i] .* σ_X[i] .- μ_X[i]
        X_subset2 = X.data[:, j] .* σ_X[j] .- μ_X[j]
        X_subset3 = X.data[:, k] .* σ_X[k] .- μ_X[k]
        data = DataFrame(y=vec(y), x1=vec(X_subset1), x2=vec(X_subset2), x3=vec(X_subset3))
        ols = lm(@formula(y ~ 1 + x1 + x2 + x3), data)
    elseif n_terms_target == 4
        i = select[1]
        j = select[2]
        k = select[3]
        l = select[4]
        X_subset1 = X.data[:, i] .* σ_X[i] .- μ_X[i]
        X_subset2 = X.data[:, j] .* σ_X[j] .- μ_X[j]
        X_subset3 = X.data[:, k] .* σ_X[k] .- μ_X[k]
        X_subset4 = X.data[:, l] .* σ_X[l] .- μ_X[l]
        data = DataFrame(y=vec(y), x1=vec(X_subset1), x2=vec(X_subset2), x3=vec(X_subset3), x4=vec(X_subset4))
        ols = lm(@formula(y ~ 1 + x1 + x2 + x3 + x4), data)

    elseif n_terms_target == 5
        i = select[1]
        j = select[2]
        k = select[3]
        l = select[4]
        m = select[5]
        X_subset1 = X.data[:, i] .* σ_X[i] .- μ_X[i]
        X_subset2 = X.data[:, j] .* σ_X[j] .- μ_X[j]
        X_subset3 = X.data[:, k] .* σ_X[k] .- μ_X[k]
        X_subset4 = X.data[:, l] .* σ_X[l] .- μ_X[l]
        X_subset5 = X.data[:, m] .* σ_X[m] .- μ_X[m]
        data = DataFrame(y=vec(y), x1=vec(X_subset1), x2=vec(X_subset2), x3=vec(X_subset3), x4=vec(X_subset4), x5=vec(X_subset5))
        ols = lm(@formula(y ~ 1+ x1 + x2 + x3 + x4 + x5), data)
    elseif n_terms_target == 6
        i = select[1]
        j = select[2]
        k = select[3]
        l = select[4]
        m = select[5]
        o = select[6]
        X_subset1 = X.data[:, i] .* σ_X[i] .- μ_X[i]
        X_subset2 = X.data[:, j] .* σ_X[j] .- μ_X[j]
        X_subset3 = X.data[:, k] .* σ_X[k] .- μ_X[k]
        X_subset4 = X.data[:, l] .* σ_X[l] .- μ_X[l]
        X_subset5 = X.data[:, m] .* σ_X[m] .- μ_X[m]
        X_subset6 = X.data[:, o] .* σ_X[o] .- μ_X[o]
        data = DataFrame(y=vec(y), x1=vec(X_subset1), x2=vec(X_subset2), x3=vec(X_subset3), x4=vec(X_subset4), x5=vec(X_subset5), x6=vec(X_subset6))
        ols = lm(@formula(y ~ 1 + x1 + x2 + x3 + x4 + x5 + x6), data)
    elseif n_terms_target == 10
        X_subset = zeros(size(X.data)[1], n_terms_target)
        for (i, sel) in enumerate(select)
            X_subset[:, i] = X.data[:, sel] .* σ_X[sel] .- μ_X[sel]
        end
        # TODO: try to allow any number of terms using schema(Term(:x1) + Term(:x2) + ..., data)
        data = DataFrame(
            y=vec(y),
            x1=vec(X_subset[:, 1]),
            x2=vec(X_subset[:, 2]),
            x3=vec(X_subset[:, 3]),
            x4=vec(X_subset[:, 4]),
            x5=vec(X_subset[:, 5]),
            x6=vec(X_subset[:, 6]),
            x7=vec(X_subset[:, 7]),
            x8=vec(X_subset[:, 8]),
            x9=vec(X_subset[:, 9]),
            x10=vec(X_subset[:, 10]))
        ols = lm(@formula(y ~ 1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10), data)
    else
        error("Unsupported number of target terms")
    end

    println(ols)
    n_samples = size(X.data)[1]
    bic = n_terms_target * log(n_samples) - 2 * loglikelihood(ols)
    println("BIC: $bic")

    return ols
end