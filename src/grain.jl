"Properties of all grains"
struct GrainProperties
    diameter
    youngs_modulus
    density
end

"Read coarse grained data"
struct CG
    name::String
    boundaries::Matrix
    buffer::Matrix
    boxes::Tuple
    path::String
    burn_in::Int
    n_dumps::Int
    shape::String
    grains::GrainProperties
end

"Generate a rectangular grid"
function rectangular_grid(boundaries, boxes)
    dx = (boundaries[2, 1] - boundaries[1, 1]) / (boxes[1] - 1)
    dy = (boundaries[2, 2] - boundaries[1, 2]) / (boxes[2] - 1)
    dz = (boundaries[2, 3] - boundaries[1, 3]) / (boxes[3] - 1)
    dt = (boundaries[2, 4] - boundaries[1, 4]) / (boxes[4] - 1)
    x = boundaries[1, 1]:dx:boundaries[2, 1]
    y = boundaries[1, 2]:dy:boundaries[2, 2]
    z = boundaries[1, 3]:dz:boundaries[2, 3]
    t = boundaries[1, 4]:dt:boundaries[2, 4]
    return x, y, z, t
end

"Generate uniform random points of [x, y, z, t] within given boundaries."
function get_sample_points(config::CG; n_samples::Int=1000)
    buffered_boundaries = config.boundaries .- config.buffer
    if config.shape == "rectangular"
        return hcat(
            rand(n_samples) .* (buffered_boundaries[2, 1] - buffered_boundaries[1, 1]) .+ buffered_boundaries[1, 1],
            rand(n_samples) .* (buffered_boundaries[2, 2] - buffered_boundaries[1, 2]) .+ buffered_boundaries[1, 2],
            rand(n_samples) .* (buffered_boundaries[2, 3] - buffered_boundaries[1, 3]) .+ buffered_boundaries[1, 3],
            rand(n_samples) .* (buffered_boundaries[2, 4] - buffered_boundaries[1, 4]) .+ buffered_boundaries[1, 4]
        )
    elseif config.shape == "cylindrical_y"
        # cylindrical about axis y. Centred at origin.
        radius_x = buffered_boundaries[2, 1] - buffered_boundaries[1, 1]
        radius_z = buffered_boundaries[2, 3] - buffered_boundaries[1, 3]
        @assert radius_x == radius_z
        r = radius_x .* rand(n_samples)
        θ = 2pi * rand(n_samples)
        return hcat(
            r .^ 0.5 .* cos.(θ),
            rand(n_samples) .* (buffered_boundaries[2, 2] - buffered_boundaries[1, 2]) .+ buffered_boundaries[1, 2],
            r .^ 0.5 .* sin.(θ),
            rand(n_samples) .* (buffered_boundaries[2, 4] - buffered_boundaries[1, 4]) .+ buffered_boundaries[1, 4]
        )
    else
        error("Unrecognised shape")
    end
end