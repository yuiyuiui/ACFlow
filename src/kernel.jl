#
# Project : Gardenia
# Source  : kernel.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2022/02/05
#

#=
```math
\begin{equation}
G(\tau) = \int^{+\infty}_{-\infty} d\omega
          \frac{e^{-\tau\omega}}{1 + e^{-\beta\omega}} A(\omega)
\end{equation}
```

```math
\begin{equation}
K(\tau,\omega) = \frac{e^{-\tau\omega}}{1 + e^{-\beta\omega}}
\end{equation}
```
=#

function build_kernel(am::AbstractMesh, fg::FermionicImaginaryTimeGrid)
    ntime = fg.ntime
    nmesh = am.nmesh
    β = fg.β

    kernel = zeros(F64, ntime, nmesh)
    for i = 1:nmesh
        for j = 1:ntime
            kernel[j,i] = exp(-fg[j] * am[i]) / (1.0 + exp(-β * am[i]))
        end
    end

    return kernel
end

#=
```math
\begin{equation}
G(i\omega_n) = \int^{+\infty}_{-\infty} d\epsilon
               \frac{1}{i\omega_n - \epsilon} A(\epsilon)
\end{equation}
```

```math
\begin{equation}
K(i\omega_n,\epsilon) = \frac{1}{i\omega_n - \epsilon}
\end{equation}
```
=#

function build_kernel(am::AbstractMesh, fg::FermionicMatsubaraGrid)
    blur = get_m("blur")
    nfreq = fg.nfreq
    nmesh = am.nmesh

    _kernel = zeros(C64, nfreq, nmesh)

    if blur > 0.0
        w_int, gaussian = gauss(blur)
        nsize = length(w_int)
        Mg = reshape(gaussian, (1, 1, nsize))
        MG = reshape(fg.ω, (nfreq, 1, 1))
        Mm = reshape(am.mesh, (1, nmesh, 1))
        Mw = reshape(w_int, (1, 1, nsize))

        integrand = Mg ./ (im * MG .- Mm .- Mw)

        for i = 1:nmesh
            for j = 1:nfreq
                _kernel[j,i] = simpson(w_int, integrand[j,i,:])
            end
        end
    else
        for i = 1:nmesh
            for j = 1:nfreq
                _kernel[j,i] = 1.0 / (im * fg[j] - am[i])
            end
        end
    end

    kernel = vcat(real(_kernel), imag(_kernel))

    return kernel
end


function build_kernel(am::AbstractMesh, bg::BosonicImaginaryTimeGrid)
end

function build_kernel(am::AbstractMesh, bg::BosonicMatsubaraGrid)
end

function build_kernel_symm(am::AbstractMesh, bg::BosonicImaginaryTimeGrid)
    ntime = bg.ntime
    nmesh = am.nmesh
    β = bg.β

    kernel = zeros(F64, ntime, nmesh)
    for i = 1:nmesh
        r = 0.5 / (1. - exp(-β * am[i]))
        for j = 1:ntime
            kernel[j,i] = r * (exp(-am[i] * bg[j]) + exp(-am[i] * (β - bg[j])))
        end
    end
    @. kernel[:,1] = 1.0

    return kernel
end

function build_kernel_symm(am::AbstractMesh, bg::BosonicMatsubaraGrid)
    blur = get_m("blur")
    nfreq = bg.nfreq
    nmesh = am.nmesh

    kernel = zeros(F64, nfreq, nmesh)

    if blur > 0.0
        w_int, gaussian = gauss(blur)
        nsize = length(w_int)
        Mg = reshape(gaussian, (1, 1, nsize))
        MG = reshape(bg.ω, (nfreq, 1, 1))
        Mm = reshape(am.mesh, (1, nmesh, 1))
        Mw = reshape(w_int, (1, 1, nsize))

        integrand_1 = Mg .* ((Mw .+ Mm) .^ 2.0) ./ ((Mw .+ Mm) .^ 2.0 .+ MG .^ 2.0)
        integrand_2 = Mg .* ((Mw .- Mm) .^ 2.0) ./ ((Mw .- Mm) .^ 2.0 .+ MG .^ 2.0)
        for j = 1:nmesh
            integrand_1[1,j,:] .= gaussian
            integrand_2[1,j,:] .= gaussian
        end
        integrand = (integrand_1 + integrand_2) / 2.0
        for i = 1:nmesh
            for j = 1:nfreq
                kernel[j,i] = simpson(w_int, integrand[j,i,:])
            end
        end
    else
        for i = 1:nmesh
            for j = 1:nfreq
                kernel[j,i] = am[i] ^ 2.0 / ( bg[j] ^ 2.0 + am[i] ^ 2.0 )
            end
        end
        if am[1] == 0.0 && bg[1] == 0.0
            kernel[1,1] = 1.0
        end
    end

    return kernel
end

function make_blur(am::AbstractMesh, A::Vector{F64}, blur::F64)
    #spl = Spline1D(am.mesh, A) # For Fermionic
    spl = Spline1D(vcat(-am.mesh[end:-1:2], am.mesh), vcat(A[end:-1:2], A)) # For bosonic
    w_int, gaussian = gauss(blur)

    nsize = length(w_int)
    nmesh = am.nmesh

    Mw = reshape(w_int, (1, nsize))
    Mg = reshape(gaussian, (1, nsize))
    Mm = reshape(am.mesh, (nmesh, 1))
    integrand = Mg .* spl.(Mm .+ Mw)

    for i = 1:nmesh
        A[i] = simpson(w_int, integrand[i,:])
    end
end

function make_singular_space(kernel::Matrix{F64})
    U, S, V = svd(kernel)
    n_svd = count(x -> x ≥ 1e-10, S)
    U_svd = U[:,1:n_svd]
    V_svd = V[:,1:n_svd]
    S_svd = S[1:n_svd]

    return U_svd, V_svd, S_svd
end

function gauss(blur::F64)
    @assert blur > 0.0
    nsize = 201
    w_int = collect(LinRange(-5.0 * blur, 5.0 * blur, nsize))
    norm = 1.0 / (blur * sqrt(2.0 * π))
    gaussian = norm * exp.(-0.5 * (w_int / blur) .^ 2.0)
    return w_int, gaussian
end
