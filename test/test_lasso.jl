using GLM, DataFrames

# Test that the function returns a vector of the same length as the number of columns in X
function test_LassoENConstrained_length()
    X = rand(5, 3)
    y = vec(rand(5))
    @test length(LassoENConstrained(X, y)) == size(X, 2)
end

# Test that the function returns the correct value when no regularization is applied
function test_LassoENConstrained_no_regularisation()
    X = [1 2 3; 5 7 11; 13 17 19; 23 29 31; 37 41 43]
    y = vec([1 2 3 4 5])
    
    data = DataFrame(y=y, X1=vec(X[:, 1]), X2=vec(X[:, 2]), X3=vec(X[:, 3]))
    ols = lm(@formula(y ~ 1 + X1 + X2 + X3), data)
    solution = GLM.coef(ols)
    permutation = [2, 3, 4, 1]
    permute!(solution, permutation)
    @test isapprox(LassoENConstrained(X, y, λ=0, γ=0, intercept=true), solution, rtol=1e-3)

    solution1 = X \ y
    @test isapprox(LassoENConstrained(X, y, λ=0, γ=0, intercept=false), solution1, rtol=1e-6)
end


@testset "LassoENConstrained.jl" begin

    test_LassoENConstrained_length()
    test_LassoENConstrained_no_regularisation()

end
