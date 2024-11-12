#
# Project : Gardenia
# Source  : rfa.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2024/09/30
#

#
# Note:
#
# The following codes for the barycentric rational function approximation
# are mostly adapted from
#
#     https://github.com/complexvariables/RationalFunctionApproximation.jl
#
# See
#
#     The AAA algorithm for rational approximation
#     Yuji Nakatsukasa, Olivier Sète, Lloyd N. Trefethen
#     SIAM Journal on Scientific Computing 40, A1494 (2018)
#
# and
#
#     AAA Rational Approximation on a Continuum
#     Tobin A. Driscoll, Yuji Nakatsukasa, Lloyd N. Trefethen
#     SIAM Journal on Scientific Computing 46, A929 (2024)
#
# for more details.
#

#=
### *Customized Structs* : *BarycentricFunction*
=#

#=
*Remarks* :

**Rational barycentric representation**

The barycentric formula takes the form of a quotient of two partial
fractions,

```math
r(z) = \frac{n(z)}{d(z)}
     = \sum^m_{j=1} \frac{w_j f_j}{z - z_j}
     {\huge/} \sum^m_{j=1} \frac{w_j}{z - z_j},
```

where ``m \ge 1`` is an integer, ``z_1, \cdots, z_m`` are a set of real
or complex distinct support points (`nodes`), ``f_1, \cdots, f_m`` are a
set of real or complex data `values`, and ``w_1, \cdots, w_m`` are a set
of real or complex `weights`. As indicated in this equation, we just let
``n(z)`` and ``d(z)`` stand for the partial fractions in the numerator
and the denominator.
=#

"""
    BarycentricFunction

Mutable struct. Barycentric representation of a rational function.

### Members
* nodes     -> Nodes of the rational function, ``z_i``.
* values    -> Values of the rational function, ``r(z_i)``.
* weights   -> Weights of the rational function, ``w_i``.
* w_times_f -> Weighted values of the rational function, ``w_i f_i``.
"""
mutable struct BarycentricFunction <: Function
    nodes     :: Vector{C64}
    values    :: Vector{C64}
    weights   :: Vector{C64}
    w_times_f :: Vector{C64}
end

"""
    BarycentricFunction(
        nodes   :: Vector{C64},
        values  :: Vector{C64},
        weights :: Vector{C64}
    )

Construct a `BarycentricFunction` type rational function.

### Arguments
* nodes   -> Interpolation nodes, ``z_i``.
* values  -> Values at the interpolation nodes, ``r(z_i)``.
* weights -> Barycentric weights, ``w_i``.

### Returns
* bf -> A BarycentricFunction struct.
"""
function BarycentricFunction(
    nodes   :: Vector{C64},
    values  :: Vector{C64},
    weights :: Vector{C64}
    )
    @assert length(nodes) == length(values) == length(weights)
    w_times_f = values .* weights
    return BarycentricFunction(nodes, values, weights, w_times_f)
end

"""
    bc_nodes(r::BarycentricFunction)

Returns the nodes of the rational interpolant `r` as a vector. Actually,
they are ``z_i``.
"""
bc_nodes(r::BarycentricFunction) = r.nodes

"""
    bc_values(r::BarycentricFunction)

Returns the nodal values of the rational interpolant `r` as a vector. They
are ``r(z_i) = f_i``.
"""
bc_values(r::BarycentricFunction) = r.values

"""
    bc_weights(r::BarycentricFunction)

Returns the weights of the rational interpolant `r` as a vector. Actually,
they are ``w_i``.
"""
bc_weights(r::BarycentricFunction) = r.weights

"""
    bc_degree(r::BarycentricFunction)

Returns the degree of the numerator and denominator of the rational `r`.
"""
bc_degree(r::BarycentricFunction) = length(r.nodes) - 1

"""
    bc_poles(r::BarycentricFunction)

Return the poles of the rational function `r`.

### Arguments
* r -> A BarycentricFunction struct.

### Returns
* pole -> List of poles.
"""
function bc_poles(r::BarycentricFunction)
    w = bc_weights(r)
    z = bc_nodes(r)
    nonzero = @. !iszero(w)
    z, w = z[nonzero], w[nonzero]
    #
    m = length(w)
    B = diagm( [zero(F64); ones(F64, m)] )
    E = [zero(F64) transpose(w); ones(F64, m) diagm(z) ];
    #
    pole = [] # Put it into scope
    try
        pole = filter( isfinite, eigvals(E, B) )
    catch
        # Generalized eigen not available in extended precision, so:
        λ = filter( z->abs(z)>1e-13, eigvals(E\B) )
        pole = 1 ./ λ
    end

    return pole
end

"""
    (r::BarycentricFunction)(z::Number)

Evaluate the barycentric rational function at `z`.

### Arguments
* z -> z ∈ ℂ.

### Returns
* val -> r(z).
"""
function (r::BarycentricFunction)(z::Number)
    if isinf(z)
        return sum(r.w_times_f) / sum(r.weights)
    end
    #
    # Try to determine whether z is a valid node
    k = findfirst(z .== r.nodes)
    #
    if isnothing(k) # Not at a node
        C = @. 1 / (z - r.nodes)
        return sum(C .* r.w_times_f) / sum(C .* r.weights)
    else            # Interpolation at node
        return r.values[k]
    end
end

#=
### *Adaptive Antoulas-Anderson Algorithm*
=#

"""
    aaa(z, y)

Adaptively compute a barycentric rational interpolant.

### Arguments
* z::AbstractVector{<:Number} -> Interpolation nodes.
* y::AbstractVector{<:Number} -> Values at nodes.
* max_degree::Integer=150 -> Maximum numerator/denominator degree to use.
* float_type::Type=F64 -> Floating point type to use for the computation.
* tol::Real=1000*eps(float_type) -> Tolerance for stopping.
* lookahead::Integer=10 -> Number of iterations to determines stagnation.
* stats::Bool=false -> Return convergence statistics.

### Returns
* r::BarycentricFunction -> The rational interpolant.
* stats::NamedTuple -> Convergence statistics, if keyword `stats = true`.

### Examples
```julia-repl
julia> z = 1im * range(-10, 10, 500);

julia> y = @. exp(z);

julia> r = aaa(z, y);

julia> bc_degree(r) # both numerator and denominator
12

julia> first(nodes(r), 4)
4-element Vector{ComplexF64}:
 0.0 - 6.272545090180361im
 0.0 + 9.43887775551102im
 0.0 - 1.1022044088176353im
 0.0 + 4.909819639278557im

julia> r(1im * π / 2)
-2.637151617496356e-15 + 1.0000000000000002im
```
"""
function aaa(
    z::AbstractVector{<:Number},
    y::AbstractVector{<:Number};
    max_degree = 150,
    float_type = F64,
    tol = 1000 * eps(float_type),
    lookahead = 10,
    stats = false
    )

    @assert float_type <: AbstractFloat
    T = float_type
    fmax = norm(y, Inf) # For scaling
    m = length(z)
    iteration = NamedTuple[]
    err = T[]
    besterr, bestidx, best = Inf, NaN, nothing

    # Allocate space for Cauchy matrix, Loewner matrix, and residual
    C = similar(z, (m, m))
    L = similar(z, (m, m))
    R = complex(zeros(size(z)))

    ȳ = sum(y) / m
    s, idx = findmax(abs(y - ȳ) for y in y)
    push!(err, s)

    # The ordering of nodes matters,
    # while the order of test points does not.
    node_index = Int[]
    push!(node_index, idx)
    test_index = Set(1:m)
    delete!(test_index, idx)

    n = 0 # Number of poles
    while true
        n += 1
        σ = view(z, node_index)
        fσ = view(y, node_index)
        # Fill in matrices for the latest node
        @inbounds @fastmath for i in test_index
            δ = z[i] - σ[n]
            # δ can be zero if there are repeats in z
            C[i, n] = iszero(δ) ? 1 / eps() : 1 / δ
            L[i, n] = (y[i] - fσ[n]) * C[i, n]
        end
        istest = collect(test_index)

        _,_,V = svd(L[istest, 1:n])
        
        w = V[:, end] # barycentric weights

        CC = view(C, istest, 1:n)
        num = CC * (w.*fσ)
        den = CC * w
        @. R[istest] = y[istest] - num / den
        push!(err, norm(R, Inf))
        push!(iteration, (; weights=w, active=copy(node_index)))

        if (Base.last(err) < besterr)
            besterr = Base.last(err)
            bestidx = length(iteration)
            best = Base.last(iteration)
        end

        # Are we done?
        if (besterr <= tol*fmax) ||
            (n == max_degree + 1) ||
            ((length(iteration) - bestidx >= lookahead) && (besterr < 1e-2*fmax))
            break
        end

        # To make sure columns of V won't be thrown away is svd and prevent overfitting
        if n>=((m+1)>>1) break end

        _, j = findmax(abs, R)
        push!(node_index, j)
        delete!(test_index, j)
        R[j] = 0
    end

    idx, w = best.active, best.weights
    r = BarycentricFunction(z[idx], y[idx], w)
    if stats
        return r, (;err, iteration)
    else
        return r
    end
end

#
# Note:
#
# The following codes for the Prony approximation are mostly adapted from
#
#     https://github.com/Green-Phys/PronyAC
#
# See
#
#     Minimal Pole Representation and Controlled Analytic Continuation
#     of Matsubara Response Functions
#     Lei Zhang and Emanuel Gull
#     arXiv:2312.10576 (2024)
#
# for more details.
#

#=
### *Customized Structs* : *PronyApproximation*
=#

#=
*Remarks* :

**Prony interpolation**

Our input data consists of an odd number ``2N + 1`` of Matsubara points
``G(i\omega_n)`` that are uniformly spaced. Prony's interpolation method
interpolates ``G_k`` as a sum of exponentials

```math
G_k = \sum^{N-1}_{i=0} w_i \gamma^k_i,
```

where ``0 \le k \le 2N``, ``w_i`` denote complex weights and ``\gamma_i``
corresponding nodes.

---

**Prony approximation**

Prony's interpolation method is unstable. We therefore employs a Prony
approximation, rather than an interpolation of ``G``. For the physical
Matsubara functions, which decay in magnitude to zero for
``i\omega_n \to i\infty``, only ``K \propto \log{1/\varepsilon}`` out of
all ``N`` nodes in the Prony approximation have weights
``|w_i| > \varepsilon``. Thus, we have

```math
\left|G_k - \sum^{K-1}_{i=0} w_i \gamma^k_i\right| \le \varepsilon,
```

for all ``0 \le k \le 2N``.
=#

"""
    PronyApproximation

Mutable struct. Prony approximation to a complex-valued Matsubara function.

### Members
* 𝑁ₚ -> Number of nodes for Prony approximation.
* ωₚ -> Non-negative Matsubara frequency.
* 𝐺ₚ -> Complex values at ωₚ.
* Γₚ -> Nodes for Prony approximation, ``γ_i``.
* Ωₚ -> Weights for Prony approximation, ``w_i``.
"""
mutable struct PronyApproximation <: Function
    𝑁ₚ :: I64
    ωₚ :: Vector{F64}
    𝐺ₚ :: Vector{C64}
    Γₚ :: Vector{C64}
    Ωₚ :: Vector{C64}
end

"""
    PronyApproximation(
        𝑁ₚ :: I64,
        ωₚ :: Vector{F64},
        𝐺ₚ :: Vector{C64},
        v  :: Vector{C64}
    )

Construct a `PronyApproximation` type interpolant function. Once it is
available, then it can be used to produce a smooth G at given ω.

This function should not be called directly by the users.

### Arguments
* 𝑁ₚ -> Number of nodes for Prony approximation.
* ωₚ -> Non-negative Matsubara frequency (postprocessed).
* 𝐺ₚ -> Complex values at ωₚ (postprocessed).
* v  -> Selected vector from the orthogonal matrix `V`.

### Returns
* pa -> A PronyApproximation struct.
"""
function PronyApproximation(
    𝑁ₚ :: I64,
    ωₚ :: Vector{F64},
    𝐺ₚ :: Vector{C64},
    v  :: Vector{C64}
    )
    # Evaluate cutoff for Γₚ
    Λ = 1.0 + 0.5 / 𝑁ₚ

    # Evaluate Γₚ and Ωₚ
    Γₚ = prony_gamma(v, Λ)
    Ωₚ = prony_omega(𝐺ₚ, Γₚ)

    # Sort Γₚ and Ωₚ
    idx_sort = sortperm(abs.(Ωₚ))
    reverse!(idx_sort)
    Ωₚ = Ωₚ[idx_sort]
    Γₚ = Γₚ[idx_sort]

    # Return a PronyApproximation struct
    return PronyApproximation(𝑁ₚ, ωₚ, 𝐺ₚ, Γₚ, Ωₚ)
end

"""
    PronyApproximation(ω₁::Vector{F64}, 𝐺₁::Vector{C64}, ε::F64)

Construct a `PronyApproximation` type interpolant function. Once it is
available, then it can be used to produce a smooth G at ω.

If the noise level of the input data is known, this function is a good
choice. The parameter `ε` can be set to the noise level.

### Arguments
* ω₁ -> Non-negative Matsubara frequency (raw values).
* 𝐺₁ -> Complex values at ωₚ (raw values).
* ε  -> Threshold for the Prony approximation.

### Returns
* pa -> A PronyApproximation struct.
"""
function PronyApproximation(ω₁::Vector{F64}, 𝐺₁::Vector{C64}, ε::F64)
    # Preprocess the input data to get the number of nodes, frequency
    # points ωₚ, and Matsubara data 𝐺ₚ.
    𝑁ₚ, ωₚ, 𝐺ₚ = prony_data(ω₁, 𝐺₁)

    # Perform singular value decomposition and select reasonable `v`.
    S, V = prony_svd(𝑁ₚ, 𝐺ₚ)
    v = prony_v(V, prony_idx(S, ε))

    return PronyApproximation(𝑁ₚ, ωₚ, 𝐺ₚ, v)
end

"""
    PronyApproximation(ω₁::Vector{F64}, 𝐺₁::Vector{C64})

Construct a `PronyApproximation` type interpolant function. Once it is
available, then it can be used to produce a smooth G at ω. Note that this
function employs a smart and iterative algorithm to determine the optimal
Prony approximation.

This function is time-consuming. But if the noise level of the input data
is unknown, this function is useful.

### Arguments
* ω₁ -> Non-negative Matsubara frequency (raw values).
* 𝐺₁ -> Complex values at ωₚ (raw values).

### Returns
* pa -> A PronyApproximation struct.
"""
function PronyApproximation(ω₁::Vector{F64}, 𝐺₁::Vector{C64})
    # Preprocess the input data to get the number of nodes, frequency
    # points ωₚ, and Matsubara data 𝐺ₚ.
    𝑁ₚ, ωₚ, 𝐺ₚ = prony_data(ω₁, 𝐺₁)

    # Perform singular value decomposition
    S, V = prony_svd(𝑁ₚ, 𝐺ₚ)

    # Next we should determine the optimal `v`
    #
    # (1) Find maximum index for the exponentially decaying region.
    exp_idx = prony_idx(S)
    #
    # (2) Find minimum index
    ε = 1000 * S[exp_idx]
    new_idx = findfirst(x -> x < ε, S)
    #
    # (3) Create lists for chosen indices and the corresponding errors.
    idxrange = range( new_idx, min(exp_idx + 10, length(S)) )
    idx_list = collect(idxrange)
    err_list = zeros(F64, length(idx_list))
    #
    # (4) Create a list of pseudo-PronyApproximations, and then evaluate
    # their reliabilities and accuracies.
    for i in eachindex(idx_list)
        idx = idx_list[i]
        #
        # Extract `v`
        v = prony_v(V, idx)
        #
        # Reproduce G using the pseudo PronyApproximation
        𝐺ₙ = PronyApproximation(𝑁ₚ, ωₚ, 𝐺ₚ, v)(ωₚ)
        #
        # Evaluate the difference and record it
        err_ave = mean(abs.(𝐺ₙ - 𝐺ₚ))
        err_list[i] = err_ave
        #
        @printf("Prony approximation %3i -> %16.12f (%16.12e)\n", i, err_ave, S[idx])
    end
    #
    # (5) Find the optimal `v`, which should minimize |𝐺ₙ - 𝐺ₚ|
    idx = idx_list[argmin(err_list)]
    v = prony_v(V, idx)
    println("The optimal Prony approximation is $idx")

    return PronyApproximation(𝑁ₚ, ωₚ, 𝐺ₚ, v)
end

"""
    prony_data(ω₁::Vector{F64}, 𝐺₁::Vector{C64})

Prepare essential data for the later Prony approximation. It will return
the number of nodes 𝑁ₚ, frequency mesh ωₚ, and Green's function data 𝐺ₚ
at this mesh.

### Arguments
* ω₁ -> Non-negative Matsubara frequency (raw values).
* 𝐺₁ -> Complex values at ωₚ (raw values), 𝐺₁ = G(ω₁).

### Returns
See above explanations.
"""
function prony_data(ω₁::Vector{F64}, 𝐺₁::Vector{C64})
    # We have to make sure the number of data points is odd.
    osize = length(ω₁)
    nsize = iseven(osize) ? osize - 1 : osize
    #
    𝑁ₚ = div(nsize, 2) # Number of nodes for Prony approximation
    ωₚ = ω₁[1:nsize]   # Matsubara frequency, ωₙ
    𝐺ₚ = 𝐺₁[1:nsize]   # Matsubara Green's function, G(iωₙ)
    #
    return 𝑁ₚ, ωₚ, 𝐺ₚ
end

"""
    prony_svd(𝑁ₚ::I64, 𝐺ₚ::Vector{C64})

Perform singular value decomposition for the matrix ℋ that is constructed
from 𝐺ₚ. It will return the singular values `S` and orthogonal matrix `V`.

### Arguments
* 𝑁ₚ -> Number of nodes.
* 𝐺ₚ -> Truncated Green's function data.

### Returns
See above explanations.

See also: [`prony_data`](@ref).
"""
function prony_svd(𝑁ₚ::I64, 𝐺ₚ::Vector{C64})
    ℋ = zeros(C64, 𝑁ₚ + 1, 𝑁ₚ + 1)
    #
    for i = 1 : 𝑁ₚ + 1
        ℋ[i,:] = 𝐺ₚ[i:i+𝑁ₚ]
    end
    #
    _, S, V = svd(ℋ)

    for i in eachindex(S)
        @printf("Singular values: %4i -> %16.12e\n", i, S[i])
    end

    return S, V
end

"""
    prony_idx(S::Vector{F64}, ε::F64)

The diagonal matrix (singular values) `S` is used to test whether the
threshold `ε` is reasonable and figure out the index for extracting `v`
from `V`.

### Arguments
* S -> Singular values of ℋ.
* ε -> Threshold provided by the users.

### Returns
* idx -> Index for S[idx] < ε.

See also: [`prony_v`](@ref) and [`prony_svd`](@ref).
"""
function prony_idx(S::Vector{F64}, ε::F64)
    # Determine idx, such that S[idx] < ε.
    idx = findfirst(x -> x < ε, S)

    # Check idx
    if isnothing(idx)
        error("Please increase ε and try again! ε ∈ [$(S[1]),$(S[end])]")
    end

    return idx
end

"""
    prony_idx(S::Vector{F64})

The diagonal matrix (singular values) `S` is used to figure out the index
for extracting `v` from `V`. This function is try to evaluate the maximum
index for the exponentially decaying region of `S`.

### Arguments
* S -> Singular values of ℋ.

### Returns
* idx -> Index for extracting `v` from `V`.

See also: [`prony_v`](@ref).
"""
function prony_idx(S::Vector{F64})
    n_max = min(3 * floor(I64, log(1.0e12)), floor(I64, 0.8 * length(S)))
    #
    idx_fit = collect(range(ceil(I64, 0.8*n_max), n_max))
    val_fit = S[idx_fit]
    𝔸 = hcat(idx_fit, ones(I64, length(idx_fit)))
    #
    𝑎, 𝑏 = pinv(𝔸) * log.(val_fit)
    𝕊 = exp.(𝑎 .* collect(range(1,n_max)) .+ 𝑏)
    #
    idx = count(S[1:n_max] .> 5.0 * 𝕊) + 1

    return idx
end

"""
    prony_v(V::Adjoint{C64, Matrix{C64}}, idx::I64)

Extract suitable vector `v` from orthogonal matrix `V` according to the
threshold `ε`.

### Arguments
* V -> Orthogonal matrix from singular value decomposition of ℋ.
* idx -> Index for extracting `v` from `V`.

### Returns
* v -> Vector `v` extracted from `V` according to `idx`.

See also: [`prony_svd`](@ref).
"""
function prony_v(V::Adjoint{C64, Matrix{C64}}, idx::I64)
    # Extract v from V
    println("Selected vector from orthogonal matrix V: ", idx)
    v = V[:,idx]

    return reverse!(v)
end

"""
    prony_gamma(v::Vector{C64}, Λ::F64)

Try to calculate Γₚ. Actually, Γₚ are eigenvalues of a matrix constructed
by `v`. `Λ` is a cutoff for Γₚ. Only those Γₚ that are smaller than `Λ`
are kept.

### Arguments
* v -> A vector extracted from `V`.
* Λ -> A cutoff for Γₚ.

### Returns
* Γₚ -> Roots of a polynominal with coefficients given in `v`.

See also: [`prony_v`](@ref).
"""
function prony_gamma(v::Vector{C64}, Λ::F64)
    # The following codes actually calculate the roots of a polynominal
    # with coefficients given in v. The roots are Γₚ.
    non_zero = findall(!iszero, v)
    trailing_zeros = length(v) - non_zero[end]
    #
    vnew = v[non_zero[1]:non_zero[end]]
    N = length(vnew)
    #
    if N > 1
        A = diagm(-1=>ones(C64, N - 2))
        @. A[1,:] = -vnew[2:end] / vnew[1]
        roots = eigvals(A)
    else
        roots = []
    end
    #
    Γₚ = vcat(roots, zeros(C64, trailing_zeros))

    filter!(x -> abs(x) < Λ, Γₚ)

    return Γₚ
end

"""
    prony_omega(𝐺ₚ::Vector{C64}, Γₚ::Vector{C64})

Try to calculate Ωₚ.

### Arguments
* 𝐺ₚ -> Complex values at ωₚ.
* Γₚ -> Nodes for Prony approximation, ``γ_i``.

### Returns
* Ωₚ -> Weights for Prony approximation, ``w_i``.
"""
function prony_omega(𝐺ₚ::Vector{C64}, Γₚ::Vector{C64})
    A = zeros(C64, length(𝐺ₚ), length(Γₚ))
    #
    for i in eachindex(𝐺ₚ)
        A[i,:] = Γₚ .^ (i - 1)
    end
    #
    return pinv(A) * 𝐺ₚ
end

"""
    (𝑝::PronyApproximation)(w::Vector{F64})

Evaluate the Prony approximation at `w`.

### Arguments
* w -> w ∈ ℝ.

### Returns
* val -> 𝑝.(w).
"""
function (𝑝::PronyApproximation)(w::Vector{F64})
    x₀ = @. (w - w[1]) / (w[end] - w[1])
    𝔸 = zeros(C64, length(x₀), length(𝑝.Ωₚ))
    #
    for i in eachindex(x₀)
        @. 𝔸[i,:] = 𝑝.Γₚ ^ (2.0 * 𝑝.𝑁ₚ * x₀[i])
    end
    #
    return 𝔸 * 𝑝.Ωₚ
end

#=
### *Customized Structs* : *BarRat Solver*
=#

"""
    BarRatContext

Mutable struct. It is used within the BarRat solver only.

### Members
* Gᵥ   -> Input data for correlator.
* grid -> Grid for input data.
* mesh -> Mesh for output spectrum.
* 𝒫    -> Prony approximation for the input data.
* ℬ    -> Barycentric rational function approximation for the input data.
* ℬP   -> It means the positions of the poles.
* ℬA   -> It means the weights / amplitudes of the poles.
"""
mutable struct BarRatContext
    Gᵥ   :: Vector{C64}
    grid :: AbstractGrid
    mesh :: AbstractMesh
    𝒫    :: Union{Missing,PronyApproximation}
    ℬ    :: Union{Missing,BarycentricFunction}
    ℬP   :: Vector{C64}
    ℬA   :: Vector{C64}
end

#=
### *Global Drivers*
=#

"""
    solve(S::BarRatSolver, rd::RawData)

Solve the analytic continuation problem by the barycentric rational
function method. This is the driver for the BarRat solver.

This solver suits Matsubara Green's functions. It supports both bosonic
and fermionic systems, irrespective of diagonal or non-diagonal functions.
It is extremely efficient. But sometimes the resulting spectral functions
could violate the sum-rules.

Similar to the StochPX and NevanAC solvers, it always returns A(ω).

### Arguments
* S -> A BarRatSolver struct.
* rd -> A RawData struct, containing raw data for input correlator.

### Returns
* mesh -> Real frequency mesh, ω.
* Aout -> Spectral function, A(ω).
* Gout -> Retarded Green's function, G(ω).
"""
function solve(S::BarRatSolver, rd::RawData)
    println("[ BarRat ]")
    #
    brc = init(S, rd)
    run(brc)
    Aout, Gout = last(brc)
    #
    return brc.mesh.mesh, Aout, Gout
end

"""
    init(S::BarRatSolver, rd::RawData)

Initialize the BarRat solver and return a BarRatContext struct.

### Arguments
* S -> A BarRatSolver struct.
* rd -> A RawData struct, containing raw data for input correlator.

### Returns
* mec -> A BarRatContext struct.
"""
function init(S::BarRatSolver, rd::RawData)
    # Prepera input data
    Gᵥ = rd.value
    println("Postprocess input data: ", length(Gᵥ), " points")

    # Prepare grid for input data
    grid = make_grid(rd)
    println("Build grid for input data: ", length(grid), " points")

    # Prepare mesh for output spectrum
    mesh = make_mesh()
    println("Build mesh for spectrum: ", length(mesh), " points")

    return BarRatContext(Gᵥ, grid, mesh, missing, missing, C64[], C64[])
end

"""
    run(brc::BarRatContext)

At first, it will try to construct a Prony approximation for the input
Matsubara data. Then the Prony approximation is used to build smooth data
set (data denoising). Finally, the barycentric rational function for this
data set is constructed. The member `ℬ` of the BarRatContext struct
(`brc`) should be updated in this function.

### Arguments
* brc -> A BarRatContext struct.

### Returns
N/A
"""
function run(brc::BarRatContext)
    # Get essential parameters
    denoise = get_r("denoise")
    ε = get_r("epsilon")

    ω = brc.grid.ω
    iω = ω * im
    G = brc.Gᵥ

    if denoise == "prony_s"
        println("Activate Prony approximation to denoise the input data")
        brc.𝒫 = PronyApproximation(ω, G, ε)
        #
        println("Construct barycentric rational function approximation")
        brc.ℬ = aaa(iω, brc.𝒫(ω))
    #
    elseif denoise == "prony_o"
        println("Activate Prony approximation to denoise the input data")
        brc.𝒫 = PronyApproximation(ω, G)
        #
        println("Construct barycentric rational function approximation")
        brc.ℬ = aaa(iω, brc.𝒫(ω))
    #
    else
        println("Construct barycentric rational function approximation")
        brc.ℬ = aaa(iω, G)
    #
    end

    get_r("atype") == "delta" && poles!(brc)
end

"""
    last(brc::BarRatContext)

It will process and write the calculated results by the BarRat solver,
including correlator at real axis, final spectral function, reproduced
correlator. The information about Prony approximation and barycentric
rational function approximation will be written as well.

### Arguments
* brc -> A BarRatContext struct.

### Returns
* Aout -> Spectral function, A(ω).
* Gout -> Retarded Green's function, G(ω).
"""
function last(brc::BarRatContext)
    # Reconstruct retarded Green's function using pole representation
    function pole_green!(_G::Vector{C64})
        η = get_r("eta")
        if η < 1.0
            # Here we should make sure that the imaginary parts of brc.ℬA
            # and brc.ℬP are quite small. Such that we can ignore them.
            rA = real(brc.ℬA)
            rP = real(brc.ℬP)
            for i in eachindex(_G)
                _G[i] = sum(@. rA / (brc.mesh.mesh[i] - rP + η * im))
            end
        else
            iA = brc.ℬA
            rP = real(brc.ℬP)
            for i in eachindex(_G)
                _G[i] = sum(@. iA / (brc.mesh.mesh[i] - rP + (η - 1) * im))
            end
        end
    end

    # By default, we should write the analytic continuation results
    # into the external files.
    _fwrite = get_b("fwrite")
    fwrite = isa(_fwrite, Missing) || _fwrite ? true : false

    # Write information about Prony approximation
    fwrite && (get_r("denoise") != "none") && begin
        write_prony(brc.𝒫.𝑁ₚ, brc.𝒫.Γₚ, brc.𝒫.Ωₚ)
        write_prony(brc.grid, brc.𝒫(brc.grid.ω))
    end

    # Write information about barycentric rational function
    fwrite && write_barycentric(brc.ℬ.nodes, brc.ℬ.values, brc.ℬ.weights)

    # Calculate full response function on real axis and write them
    _G = brc.ℬ.(brc.mesh.mesh)
    get_r("atype") == "delta" && pole_green!(_G)
    fwrite && write_complete(brc.mesh, _G)

    # Calculate and write the spectral function
    Aout = -imag.(_G) ./ π
    fwrite && write_spectrum(brc.mesh, Aout)

    # Regenerate the input data and write them
    #
    # Be careful, BarRat will always give A(ω), instead of A(ω)/ω. This
    # will lead to problem when we try to reproduce the input data, becase
    # A(ω) is not compatible with the predefined kernel. So we need to
    # convert A(ω) to A(ω)/ω when the system is bosonic.
    kernel = make_kernel(brc.mesh, brc.grid)
    if get_b("ktype") == "fermi"
        G = reprod(brc.mesh, kernel, Aout)
    else
        Aeff = Aout ./ brc.mesh.mesh
        #
        # When ω = 0.0, A(ω)/ω will produce Inf / NaN. We need to avoid this.
        @assert count(z -> isinf(z) || isnan(z), Aeff) == 1
        ind = findfirst(z -> isinf(z) || isnan(z), Aeff)
        #
        if ind == 1
            Aeff[ind] = 2.0 * Aeff[ind+1] - Aeff[ind+2]
        elseif ind == length(Aeff)
            Aeff[ind] = 2.0 * Aeff[ind-1] - Aeff[ind-2]
        else
            Aeff[ind] = (Aeff[ind-1] + Aeff[ind+1]) / 2.0
        end
        #
        G = reprod(brc.mesh, kernel, Aeff)
    end
    fwrite && write_backward(brc.grid, G)

    return Aout, _G
end

"""
    poles!(brc::BarRatContext)

Convert the barycentric rational function approximation to the classic
pole representation. Note that this feature is only suitable for the
`atype` = "delta" case. In such case, the barycenteric algorithm can find
the accurate positions for the poles via the `bc_poles()` function. But
it seems that the weights for these poles are wrong. In this function, we
just use the BFGS method to solve this optimization problem to get the
correct weights for the poles. And then the positions and weights of these
poles will be stored in `brc`, a BarRatContext struct.

### Arguments
* brc -> A BarRatContext struct.

### Returns
N/A
"""
function poles!(brc::BarRatContext)
    function 𝑓(x::Vector{C64})
        Gₙ = zeros(C64, length(brc.Gᵥ))
        iωₙ = brc.grid.ω * im
        #
        for i in eachindex(x)
            @. Gₙ = Gₙ + x[i] / (iωₙ - brc.ℬP[i])
        end
        #
        return sum(abs.(Gₙ - brc.Gᵥ))
    end

    function 𝐽!(J::Vector{C64}, x::Vector{C64})
        # The Zygote.gradient() fails here.
        J .= gradient_via_fd(𝑓, x)
    end

    # Get positions of the poles
    𝑃 = bc_poles(brc.ℬ)
    #
    # Print their positions
    println("Raw poles:")
    for i in eachindex(𝑃)
        z = 𝑃[i]
        @printf("P %4i -> %16.12f + %16.12f im \n", i, real(z), imag(z))
    end
    #
    # Filter unphysical poles
    filter!(z -> abs(imag(z)) < get_r("pcut"), 𝑃)
    if length(𝑃) == 0
        error("The number of poles is zero. You should increase pcut")
    end
    #
    # Print their positions again
    println("New poles:")
    for i in eachindex(𝑃)
        z = 𝑃[i]
        @printf("P %4i -> %16.12f + %16.12f im \n", i, real(z), imag(z))
    end
    #
    # Update BarRatContext
    brc.ℬP = 𝑃

    # Now we know positions of these poles, and we need to figure out
    # their amplitudes. This is a typical optimization problem. We just
    # employ the BFGS algorithm to do this job.
    𝐴 = zeros(C64, length(𝑃))
    res = optimize(𝑓, 𝐽!, 𝐴, max_iter = 500)
    brc.ℬA = res.minimizer
    #
    # Print their weights / amplitudes.
    println("New poles:")
    for i in eachindex(𝐴)
        z = brc.ℬA[i]
        @printf("A %4i -> %16.12f + %16.12f im \n", i, real(z), imag(z))
    end
    #
    # Well, we should check whether these amplitudes are reasonable.
    #@assert all(z -> abs(imag(z)) < get_r("pcut"), brc.ℬA)
end
