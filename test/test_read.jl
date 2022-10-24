@testset "read.jl" begin
    # Test that the function splits the data correctly
    X_all = [1 2 3; 4 5 6; 7 8 9]
    Y_all = vec([10, 20, 30])
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all)
    @test X_train == [1 2 3; 4 5 6]
    @test X_test == [7 8 9]
    @test Y_train == [10; 20;;]
    @test Y_test == [30;;]

    # Test that the function allows the split value to be specified
    X_all = [1 2 3; 4 5 6; 7 8 9]
    Y_all = vec([10, 20, 30])
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, split=0.7)
    @test X_train == [1 2 3; 4 5 6]
    @test X_test == [7 8 9]
    @test Y_train == [10; 20;;]
    @test Y_test == [30;;]
end