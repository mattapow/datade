using ScikitLearn
using MLJ
using MLJScikitLearnInterface

function train_SKL(X::Array, y; intercept=true, alphas=[0.001], l1_ratios=[1.0])
    model = ElasticNetCVRegressor(cv=3, fit_intercept=intercept, selection="random", verbose=false, l1_ratio=l1_ratios, alphas=alphas)
    X_df = DataFrame(X, :auto)
    mach = machine(model, X_df, vec(y))
    MLJ.fit!(mach)
    β = fitted_params(mach).coef
    return β
end

function train_SKL(X::SampleStatistics, y; intercept=true, alphas=[0.001], l1_ratios=[1.0])
    return train_SKL(X.data, y; intercept=intercept, alphas=alphas, l1_ratios=l1_ratios)
end
