# datade - Data to Differential Equation

The datade package allows you to learn equations in a flow using Elastic Net regression.
This technique is particularly useful for studying complex physical systems, such as granular flows, where traditional methods may not be able to uncover the underlying principles governing the behaviour of the system.

To use the datade package, you will need to load data from a simulation or experiment into the package.
The datade package currently supports data from LIGGGHTS simulations, but it can be easily adapted to support other types of data.

Once the data is loaded, you can use the stat_combiner function to transform your variables into a higher dimensional combination that is better suited for regularised regression analysis.
The goal is to uncover a linear relation in this higher dimensional space.
The train() function can then be used to run an Elastic Net regression on the transformed data, and the list_terms and get_beta_unbiased functions can be used to find the final model with unbiased coefficients.

## Installation

Download/clone the source code using `git clone`.
In a shell or Julia REPL's shell, navigate with `cd` to the datade directory.
Enter package mode by pressing ] of the REPL and type
```
(v1.0) pkg> aactivate .
(datade) pkg> instantiate
```


## Example
Here's a function demonstrating how to work with datade.
This example uses data from a LIGGGHTs simulation that must already be saved, see datade-analysis repo.
The data comes from the simulation of a coarse grained fields of a granular flow down a chute.

```
###
# Sample data from multiple chute simulations or read in if already saved
# Filter to valid observations (buffer boundaries)
# Optionally take the ensemble average over each point sample
# Perform Lasso regression
###

using LaTeXStrings, Plots, Statistics
using GLM, DataFrames
using datade
# if "using datade" doesn't work, try directly including the files

function main(
    root_experiment,
    n_samples,
    angles;
    ensembleAverage=false,
    use_convex=false,
    n_augment=0,
    importance=0.001,
    intercept=true)

    X_in, X_labels, y_in = multi_loader(root_experiment, n_samples, angles, ensembleAverage)

    # convert to SampleStatistics struct
    X_all = SampleStatistics(X_in, X_labels)

    # augment data
    if n_augment > 0
        println("Augmenting")
        for _ in 1:n_augment
            X_all = stat_combine(X_all)
        end
    end
    println("Augmented data is size: $(size(X_all.data))")

    # normalise data
    X_all, σ_X, μ_X = normalise_sigma(X_all)

    # clean data
    good_data_idx = []
    for i in 1:size(X_all.data, 1)
        if !any_bad(X_all.data[i, :])
            push!(good_data_idx, i)
        end
    end
    X_clean = SampleStatistics(X_all.data[good_data_idx, :], X_all.labels)
    println("Cleaned data is size: $(size(X_clean.data))")

    # split test from train
    X, X_test, y, y_test = train_test_split(X_clean, vec(y), split=0.75, randomise=false)
    
    # train model
    constraint_groups=[]
    β_biased = train(
        X,
        y,
        constraint_groups,
        intercept=intercept,
        use_convex=use_convex,
        start_alpha=0.004,
        alpha_high=1e1,
        alpha_low=1e-10,
        n_terms_target=1,
        log_search=false,
        importance=importance,
        )
    
    # list terms in model
    select = findall(list_terms(β_biased, X.labels; omit_final=use_convex, silent=false, importance=importance))
    println(select)
    
    # get unbiased coefficients from ols
    ols = get_beta_unbiased(y, X, σ_X, μ_X, select)
    β_unbiased = coef(ols)
    press = compute_press(β_unbiased, X_test.data[:, select], y_test)
    println("PRESS: $press")

end

root_experiment = "../Inclined_deep"
angles = [25, 25.5, 26, 26.5, 27, 27.5]
n_samples = 1000
main(root_experiment, n_samples, angles, ensembleAverage=false, n_augment=2)
```
