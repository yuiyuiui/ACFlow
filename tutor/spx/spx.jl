# ---------------------------------------------------
# smooth situation
# ---------------------------------------------------

using ACFlow
using Plots
include("../method.jl")

wn, GFV, A = spx_dfcfg_cont()

res = solve(wn, GFV)

# ---------------------------------------------------
# delta situation
# ---------------------------------------------------

using DelimitedFiles, Printf, ACFlow
using Plots, Random
import ACFlowSensitivity.generate_GFV_delta

β = 10.0;
N = 20;
Random.seed!(6)
output_bound = 5.0;
output_number = 200;
output_range = range(-output_bound, output_bound, output_number);
output_range = collect(output_range);
wn = (collect(0:(N-1)) .+ 0.5) * 2π / β;
noise0 = 0.0
noise1 = 1e-5
noise2 = 1e-4
noise3 = 1e-3

poles = [-2.0, -1.0, 1.0, 2.0]
γ_vec = 0.25 * ones(length(poles))

Gval0 = generate_GFV_delta(β, N, poles, γ_vec; noise = noise0)
Gval1 = generate_GFV_delta(β, N, poles, γ_vec; noise = noise1)
Gval2 = generate_GFV_delta(β, N, poles, γ_vec; noise = noise2)
Gval3 = generate_GFV_delta(β, N, poles, γ_vec; noise = noise3)


B = Dict{String,Any}(
    "solver" => "StochPX",  # Choose MaxEnt solver
    "mtype" => "gauss",   # Default model function
    "mesh" => "tangent", # Mesh for spectral function
    "ngrid" => N,        # Number of grid points for input data
    "nmesh" => output_number,       # Number of mesh points for output data
    "wmax" => output_bound,       # Right boundary of mesh
    "wmin" => -output_bound,      # Left boundary of mesh
    "beta" => β,      # Inverse temperature
);


S = Dict{String,Any}(
    "method" => "best",
    "nfine" => 100000,   # Number of grid points for a very fine mesh. This mesh is for the poles.
    "npole" => 4,    # Number of poles on the real axis. These poles are used to mimic the Matsubara Green's function.
    "ntry" => 1000,   # Number of attempts to figure out the solution.
    "nstep" => 100000,    #  Number of Monte Carlo sweeping steps per attempt / try.
    "theta" => 1e+6,    # . Artificial inverse temperature θ. When it is increased, the transition probabilities of Monte Carlo updates will decrease.
    "eta" => 1e-2,
);
setup_param(B, S);

mesh, reA0_delta, _ = solve(wn, Gval0)
_, reA1_delta, _ = solve(wn, Gval1)
_, reA2_delta, _ = solve(wn, Gval2)
_, reA3_delta, _ = solve(wn, Gval3)

plot(
    output_range,
    reA0_delta,
    label = "reconstruct A0(w), noise: 0.0",
    title = "SPX for delta type",
    xlabel = "w",
    ylabel = "A(w)",
)
plot!(output_range, reA1_delta, label = "reconstruct A1(w), noise: 1e-5", linewidth = 0.5)
plot!(output_range, reA2_delta, label = "reconstruct A2(w), noise: 1e-4", linewidth = 0.5)
plot!(output_range, reA3_delta, label = "reconstruct A3(w), noise: 1e-3", linewidth = 0.5)
plot!(
    poles,
    γ_vec,
    seriestype = :stem,
    linecolor = :blue,
    marker = :circle,
    markersize = 3,
    linestyle = :dash,
    label = "origin poles",
)
