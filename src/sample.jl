
using GridInterpolations, FiniteDifferences
using LinearAlgebra
using LaTeXStrings
include("SampleStatistics.jl")
include("units.jl")


eps = 1e-8

"""
Compute the tensor invariants of the Jacobian at a given query point.

The tensor is constructed using the provided functions for the three dimensions (u, v, and w) and the gradient is computed using the central finite difference method with specified order, factor, and maximum range.

Inputs:

u_func: a function for the u dimension
v_func: a function for the v dimension
w_func: a function for the w dimension
query_point: a tuple of coordinates for the query point
order: the order of the central finite difference method
factor: the factor for the central finite difference method
max_range: the maximum range for the central finite difference method
Output:

a tuple of tensor invariants for the Jacobian at the given query point
"""
function tensor_sample_jacobian(u_func, v_func, w_func, query_point, order, factor, max_range)
    tensor = zeros(3, 3)
    tensor[1, :] = grad(central_fdm(order, 1, factor=factor, max_range=max_range), u_func, query_point)[1][1:3]
    tensor[2, :] = grad(central_fdm(order, 1, factor=factor, max_range=max_range), v_func, query_point)[1][1:3]
    tensor[3, :] = grad(central_fdm(order, 1, factor=factor, max_range=max_range), w_func, query_point)[1][1:3]
    return tensor_invariants(tensor)
end


"""
tensor_sample(data, query_point, sGrid)

Compute tensor invariants of point data.
This function takes as input a tensor `data`, a query point `query_point`, and a grid `sGrid`, and it returns the tensor invariants of the data at the query point.
The tensor invariants are computed by interpolating the tensor `data` at the query point using the grid
"""
function tensor_sample(data, query_point, sGrid)
    tensor = zeros(3, 3)
    for i in 1:3
        for j in 1:3
            tensor[i, j] = GridInterpolations.interpolate(sGrid, data[i, j, :, :, :, :], query_point)
        end
    end
    out = tensor_invariants(tensor)
    return out
end

"""
field_samples(ρ::Array, v_av::Array, σ_c::Array, σ_k::Array, config::CG, sGrid; n_samples=1000, order=3, factor=1e10)

Compute the invariants of a 3x3 tensor.
If force_symmetric is true, the tensor will be symmetrized before computing the invariants.
The function returns a 1x3 array containing the three invariants.
The invariants have the same units as the tensor (not raised to a power).
"""
function tensor_invariants(tensor; force_symmetric=true)
    if force_symmetric
        tensor = 0.5 * (tensor + transpose(tensor))
    end
    tensor_diag = diag(tensor) .* Matrix{Float64}(I, 3, 3)
    off_diag = tensor .- tensor_diag
    invariant1 = tr(tensor_diag) / 3.0
    invariant2 = get_inv2(off_diag)
    invariant3 = abs(det(off_diag))^(1.0 / 3.0)
    return [invariant1, invariant2, invariant3]
end

function get_inv2(off_diag)
    invariant2 = 0.0
    for i in 1:3
        for j in 1:3
            invariant2 += off_diag[i, j] * off_diag[j, i]
        end
    end
    return (0.5 * abs(invariant2))^(0.5)
end

"""
    field_samples(ρ::Array, v_av::Array, σ_c::Array, σ_k::Array, config::CG, sGrid; n_samples=1000, order=3, factor=1e10)

TBW
"""
function field_samples(ρ::Array, v_av::Array, σ_c::Array, σ_k::Array, config::CG, sGrid; n_samples=1000, order=3, factor=1e10)
    println("Taking point samples of each field.")
    locations = get_sample_points(config, n_samples=n_samples)
    σ = σ_c .+ σ_k
    max_range = maximum(abs.(config.buffer))
    outcomes, raw_pressures = get_outcomes(locations, sGrid, σ, config)
    predictors = get_statistics(locations, sGrid, ρ, v_av, σ, σ_c, σ_k, config, order, factor, max_range, raw_pressures)

    nan_idx = findall(@. any(isnan, predictors.data))
    for idx in nan_idx
        println("NaN found in: $(predictors.labels[idx[2]])")
    end
    return predictors, outcomes, locations
end


"Return the stress invariants at each sample location."
function get_outcomes(locations, sGrid, σ, config)
    n = size(locations)[1]
    σ_out = Matrix{Float64}(undef, n, 3)
    raw_pressures = Vector{Float64}(undef, n)
    println("Getting point stress.")
    for (i, query_point) in enumerate(eachrow(locations))
        println("$i/$n")
        σ_sample = tensor_sample(σ, query_point, sGrid)

        raw_pressures[i] = σ_sample[1]
        t_i = compute_t_inertial(config.grains.diameter, config.grains.density, raw_pressures[i])
        units_here = Units(length=config.grains.diameter, density=config.grains.density, time=t_i)

        σ_out[i, :] = σ_sample ./ units_here.stress
    end
    return σ_out, raw_pressures
end
"""
Inputs:

locations: a matrix of coordinates for the locations to compute statistics
sGrid: the simulation grid
ρ: the density data from the simulation
v_av: the average velocity data from the simulation
σ: the stress tensor data from the simulation
σ_c: the cohesive stress tensor data from the simulation
σ_k: the kinetic stress tensor data from the simulation
config: the simulation configuration
order: the order of the central finite difference method for gradient calculation
factor: the factor for the central finite difference method for gradient calculation
max_range: the maximum range for the central finite difference method for gradient calculation
pressures: the pressure data for the locations
Output:

a SampleStatistics object containing the computed statistics
"""
function get_statistics(locations, sGrid, ρ, v_av, σ, σ_c, σ_k, config, order, factor, max_range, pressures)
    u = v_av[1, :, :, :, :]
    v = v_av[2, :, :, :, :]
    w = v_av[3, :, :, :, :]

    ρ_func(location) = GridInterpolations.interpolate(sGrid, ρ, location)
    u_func(location) = GridInterpolations.interpolate(sGrid, u, location)
    v_func(location) = GridInterpolations.interpolate(sGrid, v, location)
    w_func(location) = GridInterpolations.interpolate(sGrid, w, location)

    n = size(locations)[1]
    data = Matrix{Float64}(undef, n, 7)
    for (i, query_point) in enumerate(eachrow(locations))
        (σ_k1, σ_k2, σ_k3) = tensor_sample(σ_k, query_point, sGrid)
        (dv_1, dv_2, dv_3) = tensor_sample_jacobian(u_func, v_func, w_func, query_point, order, factor, max_range)

        # make variables non-dimensional point-wise.
        t_i = compute_t_inertial(config.grains.diameter, config.grains.density, pressures[i])
        units_here = Units(length=config.grains.diameter, density=config.grains.density, time=t_i)

        data[i, :] = [
            ρ_func(query_point) ./ units_here.density
            σ_k1 ./ units_here.stress
            σ_k2 ./ units_here.stress
            σ_k3 ./ units_here.stress
            dv_1 .* units_here.time
            dv_2 .* units_here.time
            dv_3 .* units_here.time
        ]
        println("$i/$n")
    end

    labels = [
        L"\phi",
        L"I^k_1", L"I^k_2", L"I^k_3",
        L"I_1", L"I_2", L"I_3",
    ]

    output = SampleStatistics(data, labels)
    @assert size(output.data)[2] == length(output.labels)
    return output
end
