#!/usr/bin/env julia

haskey(ENV,"ACFLOW_HOME") && pushfirst!(LOAD_PATH, ENV["ACFLOW_HOME"])

using DelimitedFiles
using Printf
using ACFlow

welcome()

# For MaxEnt solver

# Setup parameters
#
# For [BASE] block
# See types.jl/_PBASE for default setup
B = Dict{String,Any}(
    "finput" => "siw.inp",
    "mtype"  => "gauss",
    "mesh"   => "tangent",
    "ngrid"  => 300,
    "wmax"   => 30.0,
    "wmin"   => -30.0,
    "beta"   => 38.0,
)
#
# For [MaxEnt] block
# See types.jl/_PMaxEnt for default setup
S = Dict{String,Any}(
    "nalph"  => 13,
    "alpha"  => 1e12,
    "blur"   => 0.3,
)
#
setup_param(B, S)

# Call the solver
mesh, Aout, Gout = solve(read_data())

# Backup calculated results
cp("Aout.data", "Aout.mem1.data", force = true)
cp("Gout.data", "Gout.mem1.data", force = true)
cp("repr.data", "repr.mem1.data", force = true)
cp("Gout.data", "sigma.mem1.data", force = true)

# For MaxEnt solver

# Calculate auxiliary green's function
#
# Read self-energy function
dlm = readdlm("siw.inp")
#
# Get grid
grid = dlm[:,1]
#
# Get self-energy function
Σin = dlm[:,2] + im * dlm[:,3]
#
# Calculate auxiliary green's function
Gaux = 1.0 ./ (im * grid - Σin)
#
# Generate error bar
Gerr = fill(1e-4 + im * 1e-4, length(grid))

# Call the solver
mesh, Aout, Gout = solve(grid, Gaux, Gerr)

# Backup calculated results
cp("Aout.data", "Aout.mem2.data", force = true)
cp("Gout.data", "Gout.mem2.data", force = true)
cp("repr.data", "repr.mem2.data", force = true)

# Calculate final self-energy function on real axis
#
# Construct final self-energy function
Σout = mesh - 1.0 ./ Gout
#
# Write self-energy function
open("sigma.mem2.data", "w") do fout
    for i in eachindex(mesh)
        z = Σout[i]
        @printf(fout, "%20.16f %20.16f %20.16f\n", mesh[i], real(z), imag(z))
    end
end

# For StochOM solver

# Setup parameters
#
# For [BASE] block
# See types.jl/_PBASE for default setup
B = Dict{String,Any}(
    "solver" => "StochOM",
)
#
# For [StochOM] block
# See types.jl/_PStochOM for default setup
S = Dict{String,Any}(
    "ntry"   => 100000
)
#
setup_param(B, S, false)

# Call the solver
mesh, Aout, Gout = solve(grid, Gaux, Gerr)

# Backup calculated results
cp("Aout.data", "Aout.som.data", force = true)
cp("Gout.data", "Gout.som.data", force = true)
cp("repr.data", "repr.som.data", force = true)

# Calculate final self-energy function on real axis
#
# Construct final self-energy function
Σout = mesh - 1.0 ./ Gout
#
# Write self-energy function
open("sigma.som.data", "w") do fout
    for i in eachindex(mesh)
        z = Σout[i]
        @printf(fout, "%20.16f %20.16f %20.16f\n", mesh[i], real(z), imag(z))
    end
end
