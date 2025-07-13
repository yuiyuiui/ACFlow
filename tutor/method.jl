using LinearAlgebra, Random

function find_peaks(v, minipeak)
    idx = findall(x -> x > minipeak, v)
    diff_right = vcat(v[1:(end - 1)]-v[2:end], v[end])
    diff_left = vcat(v[1], v[2:end]-v[1:(end - 1)])
    res = []
    for j in idx
        diff_right[j] >= 0 && diff_left[j] >= 0 && push!(res, j)
    end
    return res
end

function find_peaks(mesh, v, minipeak; wind=0.01)
    @assert length(mesh) == length(v)
    n = length(mesh)
    idx = findall(x -> x > minipeak, v)
    diff_right = vcat(v[1:(end - 1)]-v[2:end], v[end])
    diff_left = vcat(v[1], v[2:end]-v[1:(end - 1)])
    tmp = []
    res = []
    for j in idx
        diff_right[j] >= 0 && diff_left[j] >= 0 && push!(tmp, j)
    end
    for j in tmp
        flag = true
        k = 1
        while j+k <= n && abs(mesh[j+k] - mesh[j]) < wind
            if v[j+k] >= v[j]
                flag = false
                break
            end
            k += 1
        end
        k = 1
        while j-k >= 1 && abs(mesh[j] - mesh[j-k]) < wind
            if v[j-k] >= v[j]
                flag = false
                break
            end
            k += 1
        end
        flag && push!(res, j)
    end
    return res
end


function integral(f::Function, a::T, b::T; h::T=T(1e-4)) where {T<:Real}
    n_raw = floor((b - a) / h)
    n = Int(n_raw)
    if isodd(n)
        n -= 1
    end
    if n < 2
        error("step is too large")
    end

    fa = f(a)
    !(typeof(fa) <: Union{T,Complex{T}}) &&
        error("Type of the output of f should be consistent with its input")
    fb = f(a + h * T(n))
    acc = fa + fb

    @inbounds for i in 1:(n-1)
        x = a + h * T(i)
        coeff = isodd(i) ? T(4) : T(2)
        acc += coeff * f(x)
    end

    return acc * (h / T(3))
end

# construct combanition of gauss waves
function continous_spectral_density(μ::Vector{T},
    σ::Vector{T},
    amplitude::Vector{T}) where {T<:Real}
    @assert length(μ) == length(σ) == length(amplitude)
    n = length(μ)
    function y(x::T)
        res = T(0)
        for i in 1:n
            res += amplitude[i] * exp(-(x - μ[i])^2 / (2 * σ[i]^2))
        end
        return res
    end
    return y
end

# generate values of G(iw_n)
function generate_GFV_cont(β::T,
    N::Int,
    A::Function;
    int_low::T=(-T(20)),
    int_up::T=T(20),
    noise::T=T(0),) where {T<:Real}
    grid = (collect(0:(N-1)) .+ 1 // 2) * T(2π) / β
    n = length(grid)
    res = zeros(Complex{T}, n)
    for i in 1:n
        res[i] = integral(x -> A(x) / (im * grid[i] - x), int_low, int_up)
    end
    for i in 1:n
        res[i] += noise * randn(T) * res[i] * exp(T(2π) * im * rand(T))
    end
    return res
end

function generate_GFV_delta(β::T,
    N::Int,
    poles::Vector{T},
    γ_vec::Vector{T};
    noise::T=T(0),) where {T<:Real}
    @assert length(poles) == length(γ_vec)
    wn = (collect(0:(N-1)) .+ 1 // 2) * T(2π) / β
    res = zeros(Complex{T}, N)
    for i in 1:N
        for j in 1:length(poles)
            res[i] += γ_vec[j] / (im * wn[i] - poles[j])
        end
    end
    for i in 1:N
        res[i] += noise * randn(T) * res[i] * exp(T(2π) * im * rand(T))
    end
    return res
end

function ssk_dfcfg_cont(;
    β=10.0,
    N=20,
    seed=6,
    μ=[0.5, -2.5],
    σ=[0.2, 0.8],
    peak=[1.0, 0.3],
    opb=5.0,
    opn=801,
    noise=0.0,
)
    Random.seed!(seed)
    A = continous_spectral_density(μ, σ, peak)
    # opr = collect(range(-opb, opb, opn))
    wn = (collect(0:(N-1)) .+ 0.5) * 2π / β
    GFV = generate_GFV_cont(β, N, A; noise=noise)


    B = Dict{String,Any}(
        "solver" => "StochSK",  # Choose MaxEnt solver
        "mesh" => "tangent", # Mesh for spectral function
        "ngrid" => N,        # Number of grid points for input data
        "nmesh" => opn,       # Number of mesh points for output data
        "wmax" => opb,       # Right boundary of mesh
        "wmin" => -opb,      # Left boundary of mesh
        "beta" => β,      # Inverse temperature
    )


    S = Dict{String,Any}(
        "method" => "chi2min",
        "nfine" => 100000,     # Number of points of a very fine linear mesh. This mesh is for the δ functions.
        "ngamm" => 1000,      # Number of δ functions. Their superposition is used to mimic the spectral functions.
        "nwarm" => 1000,      # nwarm = 1000
        "nstep" => 20000,     # Number of Monte Carlo sweeping steps.
        "ndump" => 200,       # Intervals for monitoring Monte Carlo sweeps. For every ndump steps, the StochSK solver will try to output some useful information to help diagnosis.
        "retry" => 10,        # How often to recalculate the goodness-of-fit function (it is actually χ² ) to avoid numerical deterioration.
        "theta" => 1e+6,      # Starting value for the θ parameter. The StochSK solver always starts with a huge θ parameter, and then decreases it gradually.
        "ratio" => 0.9,       # Scaling factor for the θ parameter. It should be less than 1.0.
    )
    setup_param(B, S)
    return wn, GFV, A
end

function ssk_dfcfg_delta(;
    β=10.0,
    N=20,
    seed=6,
    poles_num=2,
    opb=5.0,
    opn=801,
    noise=0.0,
)
    Random.seed!(seed)
    poles = collect(1:poles_num) .+ 0.5 * rand(poles_num)
    γ = ones(poles_num) ./ poles_num
    # opr = collect(range(-opb, opb, opn))
    wn = (collect(0:(N-1)) .+ 0.5) * 2π / β
    GFV = generate_GFV_delta(β, N, poles, γ; noise=noise)


    B = Dict{String,Any}(
        "solver" => "StochSK",  # Choose MaxEnt solver
        "mesh" => "tangent", # Mesh for spectral function
        "ngrid" => N,        # Number of grid points for input data
        "nmesh" => opn,       # Number of mesh points for output data
        "wmax" => opb,       # Right boundary of mesh
        "wmin" => -opb,      # Left boundary of mesh
        "beta" => β,      # Inverse temperature
    )


    S = Dict{String,Any}(
        "method" => "chi2min",
        "nfine" => 100000,     # Number of points of a very fine linear mesh. This mesh is for the δ functions.
        "ngamm" => poles_num,      # Number of δ functions. Their superposition is used to mimic the spectral functions.
        "nwarm" => 1000,      # nwarm = 1000
        "nstep" => 20000,     # Number of Monte Carlo sweeping steps.
        "ndump" => 200,       # Intervals for monitoring Monte Carlo sweeps. For every ndump steps, the StochSK solver will try to output some useful information to help diagnosis.
        "retry" => 10,        # How often to recalculate the goodness-of-fit function (it is actually χ² ) to avoid numerical deterioration.
        "theta" => 1e+6,      # Starting value for the θ parameter. The StochSK solver always starts with a huge θ parameter, and then decreases it gradually.
        "ratio" => 0.9,       # Scaling factor for the θ parameter. It should be less than 1.0.
    )
    setup_param(B, S)
    return wn, GFV, (poles, γ)
end

function spx_dfcfg_cont(;
    β=10.0,
    N=20,
    seed=6,
    μ=[0.5, -2.5],
    σ=[0.2, 0.8],
    peak=[1.0, 0.3],
    opb=5.0,
    opn=801,
    noise=0.0,
)
    Random.seed!(seed)
    A = continous_spectral_density(μ, σ, peak)
    # opr = collect(range(-opb, opb, opn))
    wn = (collect(0:(N-1)) .+ 0.5) * 2π / β
    GFV = generate_GFV_cont(β, N, A; noise=noise)


    B = Dict{String,Any}(
        "solver" => "StochPX",  # Choose MaxEnt solver
        "mtype" => "gauss",   # Default model function
        "mesh" => "tangent", # Mesh for spectral function
        "ngrid" => N,        # Number of grid points for input data
        "nmesh" => opn,       # Number of mesh points for output data
        "wmax" => opb,       # Right boundary of mesh
        "wmin" => -opb,      # Left boundary of mesh
        "beta" => β,      # Inverse temperature
    )


    S = Dict{String,Any}(
        "method" => "mean",
        "nfine" => 100000,   # Number of grid points for a very fine mesh. This mesh is for the poles.
        "npole" => 200,    # Number of poles on the real axis. These poles are used to mimic the Matsubara Green's function.
        "ntry" => 1000,   # Number of attempts to figure out the solution.
        "nstep" => 10000,    #  Number of Monte Carlo sweeping steps per attempt / try.
        "theta" => 1e+6,    # . Artificial inverse temperature θ. When it is increased, the transition probabilities of Monte Carlo updates will decrease.
        "eta" => 1e-4,
    )
    setup_param(B, S)

    return wn, GFV, A
end


function sac_dfcfg_cont(;
    β=10.0,
    N=20,
    seed=6,
    μ=[0.5, -2.5],
    σ=[0.2, 0.8],
    peak=[1.0, 0.3],
    opb=5.0,
    opn=801,
    noise=0.0,
)
    Random.seed!(seed)
    A = continous_spectral_density(μ, σ, peak)
    # opr = collect(range(-opb, opb, opn))
    wn = (collect(0:(N-1)) .+ 0.5) * 2π / β
    GFV = generate_GFV_cont(β, N, A; noise=noise)


    B = Dict{String,Any}(
        "solver" => "StochAC",  # Choose MaxEnt solver
        "mesh" => "tangent", # Mesh for spectral function
        "ngrid" => N,        # Number of grid points for input data
        "nmesh" => opn,       # Number of mesh points for output data
        "wmax" => opb,       # Right boundary of mesh
        "wmin" => -opb,      # Left boundary of mesh
        "beta" => β,      # Inverse temperature
    )


    S = Dict{String,Any}(
        "nfine" => 10000,   # Number of grid points for a very fine mesh. This mesh is for the poles.
        "ngamm" => 512,   # Number of δ functions. Their superposition is used to mimic the spectral functions.
        "nwarm" => 4000,   # Number of Monte Carlo thermalization steps.
        "nstep" => 4000000,    #  Number of Monte Carlo steps.
        "ndump" => 40000,    #  Intervals for monitoring Monte Carlo sweeps. For every ndump steps, the StochAC solver will try to output some useful information to help diagnosis.
        "nalph" => 20,      # Total number of the α parameters.
        "alpha" => 1.0,      # Starting value for the α parameter. The StochAC solver always starts with a small α parameter, and then increases it gradually.
        "ratio" => 1.2,       # Scaling factor for the α parameter. It should be larger than 1.0.
    )
    setup_param(B, S)

    return wn, GFV, A
end


function sac_dfcfg_delta(;
    β=10.0,
    N=20,
    seed=6,
    poles_num=2,
    opb=5.0,
    opn=801,
    noise=0.0,
)
    Random.seed!(seed)
    poles = collect(1:poles_num) .+ 0.5 * rand(poles_num)
    γ = ones(poles_num) ./ poles_num
    wn = (collect(0:(N-1)) .+ 0.5) * 2π / β
    GFV = generate_GFV_delta(β, N, poles, γ; noise=noise)

    B = Dict{String,Any}(
        "solver" => "StochAC",  # Choose MaxEnt solver
        "mesh" => "tangent", # Mesh for spectral function
        "ngrid" => N,        # Number of grid points for input data
        "nmesh" => opn,       # Number of mesh points for output data
        "wmax" => opb,       # Right boundary of mesh
        "wmin" => -opb,      # Left boundary of mesh
        "beta" => β,      # Inverse temperature
    )


    S = Dict{String,Any}(
        "nfine" => 10000,   # Number of grid points for a very fine mesh. This mesh is for the poles.
        "ngamm" => poles_num,   # Number of δ functions. Their superposition is used to mimic the spectral functions.
        "nwarm" => 4000,   # Number of Monte Carlo thermalization steps.
        "nstep" => 4000000,    #  Number of Monte Carlo steps.
        "ndump" => 40000,    #  Intervals for monitoring Monte Carlo sweeps. For every ndump steps, the StochAC solver will try to output some useful information to help diagnosis.
        "nalph" => 20,      # Total number of the α parameters.
        "alpha" => 1.0,      # Starting value for the α parameter. The StochAC solver always starts with a small α parameter, and then increases it gradually.
        "ratio" => 1.2,       # Scaling factor for the α parameter. It should be larger than 1.0.
    )
    setup_param(B, S)

    return wn, GFV, (poles, γ)
end


function som_dfcfg_cont(;
    β=10.0,
    N=20,
    seed=6,
    μ=[0.5, -2.5],
    σ=[0.2, 0.8],
    peak=[1.0, 0.3],
    opb=5.0,
    opn=801,
    noise=0.0,
)
    Random.seed!(seed)
    A = continous_spectral_density(μ, σ, peak)
    # opr = collect(range(-opb, opb, opn))
    wn = (collect(0:(N-1)) .+ 0.5) * 2π / β
    GFV = generate_GFV_cont(β, N, A; noise=noise)


    B = Dict{String,Any}(
        "solver" => "StochOM",  # Choose MaxEnt solver
        "mesh" => "tangent", # Mesh for spectral function
        "ngrid" => N,        # Number of grid points for input data
        "nmesh" => opn,       # Number of mesh points for output data
        "wmax" => opb,       # Right boundary of mesh
        "wmin" => -opb,      # Left boundary of mesh
        "beta" => β,      # Inverse temperature
    )


    S = Dict{String,Any}(
        "ntry" => 2000,    # Number of attempts to figure out the solution.
        "nstep" => 1000,   # Number of Monte Carlo steps per try.
        "nbox" => 100,     # Number of boxes. Their superposition is used to construct the spectral functions.
        "sbox" => 0.005,   # Minimum area of the randomly generated boxes.
        "wbox" => 0.02,    # Minimum width of the randomly generated boxes.
        "norm" => -1.0,    # Is the norm calculated?
    )
    setup_param(B, S)

    return wn, GFV, A
end


function som_dfcfg_delta(;
    β=10.0,
    N=20,
    seed=6,
    poles_num=2,
    opb=5.0,
    opn=801,
    noise=0.0,
)
    Random.seed!(seed)
    poles = collect(1:poles_num) .+ 0.5 * rand(poles_num)
    γ = ones(poles_num) ./ poles_num
    wn = (collect(0:(N-1)) .+ 0.5) * 2π / β
    GFV = generate_GFV_delta(β, N, poles, γ; noise=noise)

    B = Dict{String,Any}(
        "solver" => "StochOM",  # Choose MaxEnt solver
        "mesh" => "tangent", # Mesh for spectral function
        "ngrid" => N,        # Number of grid points for input data
        "nmesh" => opn,       # Number of mesh points for output data
        "wmax" => opb,       # Right boundary of mesh
        "wmin" => -opb,      # Left boundary of mesh
        "beta" => β,      # Inverse temperature
    )


    S = Dict{String,Any}(
        "ntry" => 2000,    # Number of attempts to figure out the solution.
        "nstep" => 1000,   # Number of Monte Carlo steps per try.
        "nbox" => 100,     # Number of boxes. Their superposition is used to construct the spectral functions.
        "sbox" => 0.005,   # Minimum area of the randomly generated boxes.
        "wbox" => 0.02,    # Minimum width of the randomly generated boxes.
        "norm" => -1.0,    # Is the norm calculated?
    )
    setup_param(B, S)

    return wn, GFV, (poles, γ)
end