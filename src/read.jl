using NPZ
using FileIO, JLD2, JSON
using Random

include("grain.jl")
include("SampleStatistics.jl")
include("sample.jl")

eps = 1e-16

"""
function cgread(config)
    cgread(config)
    Reads in data from a compressed numpy file (.npz) and returns the data as arrays of densities, velocities, and stresses.
    
    Inputs
    config: a dictionary containing the path to the .npz file and the burn-in value for the data.
    Outputs
    Four arrays containing the densities, velocities, and stresses from the input data.
    
    Example
    config = Dict("path"=>"mydata.npz", "burn_in"=>500)
    ρ, v_av, σ_c, σ_k = cgread(config)
    ρ, v_av, σ_c, and σ_k will contain the densities, velocities, and stresses from the input .npz file, with the first 500 time steps removed (assuming a burn-in value of 500).

"""
function cgread(config)
    println("Reading in data from numpy file.")
    data_in = npzread(config.path)
    ρ = data_in["RHO"][:, :, :, config.burn_in:end]
    v_av = data_in["VAVG"][:, :, :, :, config.burn_in:end]
    σ_c = data_in["TC"][:, :, :, :, :, config.burn_in:end]
    σ_k = data_in["TK"][:, :, :, :, :, config.burn_in:end]
    return ρ, v_av, σ_c, σ_k
end

"""
The cgread_prep function takes in a geometry parameter and returns a CG type object with various properties based on the geometry parameter.
The geometry parameter can be one of the following: "Mill", "Inclined", "Inclined_deep", "Inclined_more_gravity", or "Inclined_deep/theta" followed by a number.
The returned CG object contains the following properties: name, boundaries, buffer, boxes, grains, burn_in, n_dumps, dump_dt, path, and shape.
"""
function cgread_prep(geometry)
    #TODO: turn these into JSON files
    valid_geometry = false
    if geometry in ("Mill", "Inclined", "Inclined_deep", "Inclined_more_gravity")
        valid_geometry = true
    elseif valid_geometry || occursin("Inclined_deep/theta", geometry)
        # E.G. Inclined_deep/theta19
        valid_geometry = true
    elseif valid_geometry || occursin("Mill/rot", geometry)
        valid_geometry = true
    end
    @assert valid_geometry
    if occursin("Mill", geometry)
        name = "Mill"
        burn_in = 400
        n_dumps = 1000
        grains = GrainProperties(0.1, 5e8, 2500.0)
        dump_dt = 0.00001 * 1000
        boxes = (30, 10, 30, n_dumps - burn_in + 1)
        boundaries = [
            [-3 / sqrt(2) -1 -3 / sqrt(2) burn_in * dump_dt]
            [3 / sqrt(2) 1 3 / sqrt(2) dump_dt * n_dumps]
        ]
        buffer = [[-0.002 -0.002 -0.002 0.0]; [0.002 0.002 0.002 0.0]]
        path = geometry * "/CG/CG_SAG.npz"
        shape = "cylindrical_y"
    elseif geometry == "Inclined"
        name = "Inclined"
        grains = GrainProperties(0.001, 5e6, 2500.0)
        burn_in = 1
        n_dumps = 700
        dump_dt = 0.00001 * 1000
        boxes = (10, 10, 10, n_dumps - burn_in + 1)
        buffer = [[-0.001 -0.001 -0.01 0.0]; [0.001 0.001 0.007 0.0]]
        boundaries = [
            [-0.01 -0.01 0.0 burn_in * dump_dt]
            [0.01 0.01 0.02 dump_dt * n_dumps]
        ]
        path = "../Inclined/CG/CG_inclined.npz"
        shape = "rectangular"
    elseif geometry == "Inclined_more_gravity"
        name = "Inclined"
        grains = GrainProperties(0.001, 5e6, 2500.0)
        burn_in = 200
        n_dumps = 500
        dump_dt = 0.00001 * 1000
        boxes = (10, 10, 10, n_dumps - burn_in + 1)
        buffer = [[-0.001 -0.001 -0.001 0.0]; [0.001 0.001 0.007 0.0]]
        boundaries = [
            [-0.01 -0.01 0.0 burn_in * dump_dt]
            [0.01 0.01 0.02 dump_dt * n_dumps]
        ]
        path = "../Inclined_more_gravity/CG/CG_inclined.npz"
        shape = "rectangular"
    elseif geometry == "Inclined_deep" || occursin("Inclined_deep/theta", geometry)
        name = "Inclined"
        grains = GrainProperties(0.001, 5e6, 2500.0)
        burn_in = 1800
        n_dumps = 3300
        dump_dt = 0.00001 * 1000
        boxes = (10, 10, 20, n_dumps - burn_in + 1)
        buffer = [[-0.001 -0.001 -0.01 0.0]; [0.001 0.001 0.007 0.0]]
        boundaries = [
            [-0.01 -0.01 0.0 burn_in * dump_dt]
            [0.01 0.01 0.04 dump_dt * n_dumps]
        ]
        path = "../Inclined_deep/CG/CG_inclined.npz"
        if occursin("Inclined_deep/theta", geometry)
            path = geometry * "/CG/CG_inclined.npz"
        end
        shape = "rectangular"
    else
        error("No valid geometry picked.")
    end
    return CG(name, boundaries, buffer, boxes, path, burn_in, n_dumps, shape, grains)
end

"This function reads in and processes sample data for a given geometry.
It takes in the name of the geometry as an input and returns the sample data, its minimum and maximum values, a group for mass continuity, the response data, and the locations of the samples.
The number of samples can be specified using the optional input n_samples."
function read_sample(geometry; n_samples=1000)
    # read in data
    config = cgread_prep(geometry)
    ρ, v_av, σ_c, σ_k = cgread(config)

    # generate sample points
    x, y, z, t = rectangular_grid(config.boundaries, config.boxes)
    sGrid = SimplexGrid(x, y, z, t)
    X, outcomes, locations = field_samples(ρ, v_av, σ_c, σ_k, config, sGrid, n_samples=n_samples)

    # TODO: momentum and energy groups
    # mass_continuity_group = (5, [1, 7], [6, 2], [1, 13], [11, 3], [1, 19], [16, 4])
    mass_continuity_group = []

    return X, mass_continuity_group, outcomes, locations
end

"Split the data into a test and train set."
function train_test_split(X_all::Array, Y_all::Vector; split=0.8, randomise=true)
    if randomise
        p = randperm(Random.seed!(1), size(X_all, 1))
        for j in 1:size(X_all, 2)
            permute!(X_all[:, j], p)
        end
        permute!(Y_all, p)
    end

    m = size(X_all)[1]
    n_split = Int(floor((m * split)))
    ind_train = 1:n_split
    ind_test = n_split+1:m
    X_train = X_all[ind_train, :]
    X_test = X_all[ind_test, :]
    Y_train = Y_all[ind_train, :]
    Y_test = Y_all[ind_test, :]
    return X_train, X_test, Y_train, Y_test
end

"Split the data into a test and train set."
function train_test_split(X_all::SampleStatistics, Y_all::Vector; split=0.8, randomise=true)
    X_train_, X_test_, Y_train, Y_test = train_test_split(X_all.data, Y_all; split=split, randomise=randomise)
    X_train = SampleStatistics(X_train_, X_all.labels)
    X_test = SampleStatistics(X_test_, X_all.labels)
    return X_train, X_test, Y_train, Y_test
end

"""
Load sample data for a given experiment.

Inputs:

experiment: the name of the experiment
n_samples: (optional) the number of samples to load, default is 1000
Outputs:

X_all: a SampleStatistics object of input data for the samples
groups: a vector of grouped constraints for the optimisation
Y_all: a vector of outcomes data for the samples. The three invariants of the stress tensor.
locations: a matrix of coordinates for the locations of the samples
"""
function load_sample(experiment; n_samples=1000)
    load_filename = experiment * "/samples/n" * string(n_samples) * ".jld2"
    X_all,groups, Y_all, locations = (nothing, nothing, nothing, nothing)
    try
        X_all,groups, Y_all, locations = FileIO.load(load_filename, "X_all", "groups", "Y_all", "locations")
        println("Samples loaded.")
    catch
        println("Load failed. Sampling.")
        X_all, groups, Y_all, locations = read_sample(experiment, n_samples=n_samples)
        data_dict = Dict(
            "X_all" => X_all,
            "groups" => groups,
            "Y_all" => Y_all,
            "locations" => locations,
        )
        println("Saving to $load_filename")
        save(load_filename, data_dict)
        println("Saved. ")
    end
    return X_all, groups, Y_all, locations
end

"Remove data row if Y is greater than the max value"
function remove_outliers(X_all, Y_all, max_val)
    indices = findall(Y_all .> max_val)

    n_samples = length(Y_all) - length(indices)
    n_cols = size(X_all.data)[2]

    X_out_mat = zeros((n_samples, n_cols))
    Y_out = zeros(n_samples)

    k = 1
    for i in 1:n_samples
        if i ∉ indices
            Y_out[k] = Y_all[i]
            X_out_mat[k, :] = X_all.data[i, :]
            k += 1
        end
    end

    X_all_out = SampleStatistics(X_out_mat, X_all.labels)
    return X_all_out, Y_all
end

function multi_loader(root_experiment, n_samples, angles, ensembleAverage)
    n_theta = length(angles)
    n_vars = 7
    if ensembleAverage
        y_in = zeros(n_theta)
        I = zeros(n_theta)
        ϕ = zeros(n_theta)
        I_k3 = zeros(n_theta)
        I_k1_half = zeros(n_theta)
        I_3 = zeros(n_theta)
        X_in = zeros((n_theta, n_vars))
        y_in = zeros(n_theta)
    else
        I = zeros(n_samples, n_theta)
        ϕ = zeros(n_samples, n_theta)
        I_k3 = zeros(n_samples, n_theta)
        I_k1_half = zeros(n_samples, n_theta)
        I_3 = zeros(n_samples, n_theta)
        X_in = zeros((n_samples*n_theta, n_vars))
        y_in = zeros(n_samples*n_theta)
    end
        
    X_labels = nothing

    for (i_theta, theta) in enumerate(angles)
        if root_experiment == "../Mill"
            path_experiment = root_experiment * "/rot" * theta
        else
            theta_parsed = theta
            if theta % 1 == 0
                theta_parsed = Int(theta)
            end
            path_experiment = root_experiment * "/theta" * string(theta_parsed)
        end

        if ensembleAverage
            X_, y_, X_labels = get_mean_sample(path_experiment, n_samples)
            y_in[i_theta] = y_[2]  # second invariant of traceless part of stress tensor
            X_in[i_theta, :] = X_
        else
            X_, y_, X_labels = get_all_sample(path_experiment, n_samples)
            # y[:, i_theta] = y_[:, 2]  # second invariant of traceless part of stress tensor
            # I[:, i_theta] = X_[:, 6]
            # I_3[:, i_theta] = X_[:, 7]
            # ϕ[:, i_theta] = X_[:, 1]
            # I_k1[:, i_theta] = X_[:, 2]
            # # # I_k1_half[:, i_theta] = X_[:, 17]
            i_range = (i_theta - 1)*n_samples+1:(i_theta)*n_samples
            X_in[i_range, :] = X_
            y_in[i_range] = y_[:, 2]
        end
    end
    return X_in, X_labels, y_in
end

function get_mean_sample(experiment, n_samples)
    X_all, groups, σ, locations = load_sample(experiment, n_samples=n_samples)
    σ_mean = mean(σ, dims=1)
    X_mean = mean(X_all.data, dims=1)
    return X_mean, σ_mean, X_all.labels
end

function get_all_sample(experiment, n_samples)
    X_all, groups, σ, locations = load_sample(experiment, n_samples=n_samples)
    return X_all.data, σ, X_all.labels
end
