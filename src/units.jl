"Store common units."
struct Units{T}
    length::T
    mass::T
    time::T

    density::T
    stress::T
    velocity::T
end

# "Define unit values from length, mass and time."
# Units(; length, mass, time) = Units(
#     length,
#     mass,
#     time,
#     mass / length^3,
#     mass / length / time^2,
#     length / time)

"Define unit values from length, density and time."
Units(; length, density, time) = Units(
    length,
    density * length^3,
    time,
    density,
    density * length^2 / time^2,
    length / time)

function compute_t_inertial(diameter, ρ_g, pressure)
    if pressure == 0.0
        pressure = eps
    elseif pressure < 0.0
        pressure = abs(pressure)
    end
    return diameter * sqrt(ρ_g / pressure)
end

function nondimensionalise!(units, ρ, v_av, σ_c, σ_k, boundaries)
    ρ ./= units.density
    v_av ./= units.velocity
    σ_c ./= units.stress
    σ_k ./= units.stress
    boundaries[:, 1:3] ./= units.length
    boundaries[:, end] ./= units.time
    return nothing
end
