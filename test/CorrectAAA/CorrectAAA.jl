using ACFlow, Plots, DelimitedFiles, QuadGK, Distributions, Random

# construct combanition of gauss waves
function continous_spectral_density(
    μ::Vector{Float64},
    σ::Vector{Float64},
    amplitude::Vector{Float64},
)
    @assert length(μ) == length(σ) == length(amplitude)
    n = length(μ)
    function y(x::Float64)
        res = 0
        for i = 1:n
            res += amplitude[i] * exp(-(x - μ[i])^2 / (2 * σ[i]^2))
        end
        return res
    end
    return y
end

# generate values of G(iw_n)
function generate_G_values_cont(
    β::Float64,
    N::Int64,
    A;
    int_low::Float64 = -20.0,
    int_up::Float64 = 20.0,
    noise::Float64 = 0.0,
)
    grid = (collect(0:(N-1)) .+ 0.5) * 2π / β
    n = length(grid)
    res = zeros(ComplexF64, n)
    for i = 1:n
        res[i] = quadgk(x -> A(x) / (im * grid[i] - x), int_low, int_up)[1]
    end
    NL = Normal(0.0, 1.0)   # Normal list
    for i = 1:n
        res[i] += noise * rand(NL) * res[i] * exp(2π * im * rand())
    end
    return res
end

function correct_aaa(;
    seed = 6,
    μ = [0.5, -2.5],
    σ = [0.2, 0.8],
    amplitude = [1.0, 0.3],
    β = 10.0,
    N = 20,
    output_bound = 8.0,
    output_number = 801,
    noise = 1e-5,
)

    Random.seed!(seed)
    iwn = collect((0:(N-1)) .+ 0.5) * 2π / β
    A = continous_spectral_density(μ, σ, amplitude)
    GFV = generate_G_values_cont(β, N, A; noise = noise)

    B = Dict{String,Any}(
        "solver" => "BarRat",  # Choose MaxEnt solver
        "mtype" => "gauss",   # Default model function
        "mesh" => "tangent", # Mesh for spectral function
        "ngrid" => N,        # Number of grid points for input data
        "nmesh" => output_number,       # Number of mesh points for output data
        "wmax" => output_bound,       # Right boundary of mesh
        "wmin" => -output_bound,      # Left boundary of mesh
        "beta" => β,      # Inverse temperature
    )

    S = Dict{String,Any}(
        "atype" => "cont",
        #"denoise"=>"prony_o",
        "denoise" => "none",
        #"denoise"=>"prony_s",
        "epsilon" => 1e-10,
        "pcut" => 1e-3,
        "eta" => 1e-2,
    )
    setup_param(B, S)

    mesh, reA, _ = solve(iwn, GFV)
    return mesh, reA, A.(mesh)
end

mesh, reA, originA = correct_aaa()

plot(mesh, reA, label = "reconstruct spectral density")
plot!(mesh, originA, label = "origin spectral density")
