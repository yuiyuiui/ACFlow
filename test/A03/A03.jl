#!/usr/bin/env julia

push!(LOAD_PATH, ENV["ACFLOW_HOME"])

using ACFlow

welcome()

# For MaxEnt solver

# Setup parameters
C = Dict{String,Any}(
    "finput" => "sigma.inp",
    "mtype"  => "2lorentz",
    "ngrid"  => 100,
    "wmax"   => 20.0,
    "wmin"   => -20.0,
    "beta"   => 5.0,
    "pmodel" => [2.0, -5.0, 5.0],
)
#
S = Dict{String,Any}(
    "nalph"  => 15,
    "alpha"  => 1e12,
)
#
setup_param(C, S)

# Call the solver
mesh, Aout, Gout = solve(read_data())

# Backup calculated results
cp("Aout.data", "Aout.mem.data", force = true)
cp("Gout.data", "Gout.mem.data", force = true)
cp("repr.data", "repr.mem.data", force = true)

# For StochOM solver

# Setup parameters
C = Dict{String,Any}(
    "solver" => "StochOM"
)
#
S = Dict{String,Any}(
    "ntry"   => 100000
)
#
setup_param(C, S, false)

# Call the solver
mesh, Aout, Gout = solve(read_data())

# Backup calculated results
cp("Aout.data", "Aout.som.data", force = true)
cp("Gout.data", "Gout.som.data", force = true)
cp("repr.data", "repr.som.data", force = true)
