#
# Conjugate gradient line search implementation from:
#   W. W. Hager and H. Zhang (2006) Algorithm 851: CG_DESCENT, a
#     conjugate gradient method with guaranteed descent. ACM
#     Transactions on Mathematical Software 32: 113–137.
#

mutable struct LineSearchException{T<:Real} <: Exception
    message::AbstractString
    alpha::T
end

function LS(state::BFGSState, alpha::T, scaled::Bool) where T
    PT = promote_type(T, real(eltype(state.ls)))
    if scaled == true && (ns = real(norm(state.ls))) > convert(PT, 0)
        state.alpha = convert(PT, min(alpha, ns)) / ns
    else
        state.alpha = convert(PT, alpha)
    end
end

function LS(df::BFGSDifferentiable, x::AbstractArray,
            s::AbstractArray, c::Real, phi_0::Real, dphi_0::Real)
    delta = 0.1
    sigma = 0.9
    alphamax = Inf
    rho = 5.0
    epsilon = 1e-6
    gamma = 0.66
    linesearchmax = 50
    psi3 = 0.1
    mayterminate = Ref{Bool}(false)

    ϕdϕ = make_ϕdϕ(df, similar(x), x, s)

    T = typeof(c)
    zeroT = convert(T, 0)
    if !(isfinite(phi_0) && isfinite(dphi_0))
        throw(LineSearchException("Value and slope at step length = 0 must be finite.", T(0)))
    end
    if dphi_0 >= eps(T) * abs(phi_0)
        throw(LineSearchException("Search direction is not a direction of descent.", T(0)))
    elseif dphi_0 >= 0
        return zeroT, phi_0
    end

    # Prevent values of x_new = x+αs that are likely to make
    # ϕ(x_new) infinite
    iterfinitemax::Int = ceil(Int, -log2(eps(T)))
    alphas = [zeroT] # for bisection
    values = [phi_0]
    slopes = [dphi_0]

    phi_lim = phi_0 + epsilon * abs(phi_0)
    @assert c >= 0
    c <= eps(T) && return zeroT, phi_0
    @assert isfinite(c) && c <= alphamax
    phi_c, dphi_c = ϕdϕ(c)
    iterfinite = 1
    while !(isfinite(phi_c) && isfinite(dphi_c)) && iterfinite < iterfinitemax
        mayterminate[] = false
        iterfinite += 1
        c *= psi3
        phi_c, dphi_c = ϕdϕ(c)
    end
    if !(isfinite(phi_c) && isfinite(dphi_c))
        @warn("Failed to achieve finite new evaluation point, using alpha=0")
        mayterminate[] = false # reset in case another initial guess is used next
        return zeroT, phi_0
    end
    push!(alphas, c)
    push!(values, phi_c)
    push!(slopes, dphi_c)

    # If c was generated by quadratic interpolation, check whether it
    # satisfies the Wolfe conditions
    if mayterminate[] &&
          satisfies_wolfe(c, phi_c, dphi_c, phi_0, dphi_0, phi_lim, delta, sigma)
        mayterminate[] = false # reset in case another initial guess is used next
        return c, phi_c # phi_c
    end
    # Initial bracketing step (HZ, stages B0-B3)
    isbracketed = false
    ia = 1
    ib = 2
    @assert length(alphas) == 2
    iter = 1
    cold = -one(T)
    while !isbracketed && iter < linesearchmax
        if dphi_c >= zeroT
            # We've reached the upward slope, so we have b; examine
            # previous values to find a
            ib = length(alphas)
            for i = (ib - 1):-1:1
                if values[i] <= phi_lim
                    ia = i
                    break
                end
            end
            isbracketed = true
        elseif values[end] > phi_lim
            # The value is higher, but the slope is downward, so we must
            # have crested over the peak. Use bisection.
            ib = length(alphas)
            ia = 1
            if c ≉  alphas[ib] || slopes[ib] >= zeroT
                error("c = ", c)
            end
            # ia, ib = bisect(phi, lsr, ia, ib, phi_lim) # TODO: Pass options
            ia, ib = ls_bisect!(ϕdϕ, alphas, values, slopes, ia, ib, phi_lim)
            isbracketed = true
        else
            # We'll still going downhill, expand the interval and try again.
            # Reaching this branch means that dphi_c < 0 and phi_c <= phi_0 + ϵ_k
            # So cold = c has a lower objective than phi_0 up to epsilon. 
            # This makes it a viable step to return if bracketing fails.

            # Bracketing can fail if no cold < c <= alphamax can be found with finite phi_c and dphi_c.
            # Going back to the loop with c = cold will only result in infinite cycling.
            # So returning (cold, phi_cold) and exiting the line search is the best move.
            cold = c
            phi_cold = phi_c
            if nextfloat(cold) >= alphamax
                mayterminate[] = false # reset in case another initial guess is used next
                return cold, phi_cold
            end
            c *= rho
            if c > alphamax
                c = alphamax
            end
            phi_c, dphi_c = ϕdϕ(c)
            iterfinite = 1
            while !(isfinite(phi_c) && isfinite(dphi_c)) && c > nextfloat(cold) && iterfinite < iterfinitemax
                alphamax = c # shrinks alphamax, assumes that steps >= c can never have finite phi_c and dphi_c
                iterfinite += 1
                c = (cold + c) / 2
                phi_c, dphi_c = ϕdϕ(c)
            end
            if !(isfinite(phi_c) && isfinite(dphi_c))
                return cold, phi_cold
            end
            push!(alphas, c)
            push!(values, phi_c)
            push!(slopes, dphi_c)
        end
        iter += 1
    end
    while iter < linesearchmax
        a = alphas[ia]
        b = alphas[ib]
        @assert b > a
        if b - a <= eps(b)
            mayterminate[] = false # reset in case another initial guess is used next
            return a, values[ia] # lsr.value[ia]
        end
        iswolfe, iA, iB = ls_secant2!(ϕdϕ, alphas, values, slopes, ia, ib, phi_lim, delta, sigma)
        if iswolfe
            mayterminate[] = false # reset in case another initial guess is used next
            return alphas[iA], values[iA] # lsr.value[iA]
        end
        A = alphas[iA]
        B = alphas[iB]
        @assert B > A
        if B - A < gamma * (b - a)
            if nextfloat(values[ia]) >= values[ib] && nextfloat(values[iA]) >= values[iB]
                # It's so flat, secant didn't do anything useful, time to quit
                mayterminate[] = false # reset in case another initial guess is used next
                return A, values[iA]
            end
            ia = iA
            ib = iB
        else
            # Secant is converging too slowly, use bisection
            c = (A + B) / convert(T, 2)

            phi_c, dphi_c = ϕdϕ(c)
            @assert isfinite(phi_c) && isfinite(dphi_c)
            push!(alphas, c)
            push!(values, phi_c)
            push!(slopes, dphi_c)

            ia, ib = ls_update!(ϕdϕ, alphas, values, slopes, iA, iB, length(alphas), phi_lim)
        end
        iter += 1
    end

    throw(LineSearchException("Linesearch failed to converge, reached maximum iterations $(linesearchmax).",
                              alphas[ia]))
end

function make_ϕdϕ(df::BFGSDifferentiable, x_new, x, s)
    function ϕdϕ(α)
        # Move a distance of alpha in the direction of s
        x_new .= x .+ α.*s

        # Evaluate ∇f(x+α*s)
        value_gradient!(df, x_new)

        # Calculate ϕ'(a_i)
        value(df), real(dot(gradient(df), s))
    end
    ϕdϕ
end

# Check Wolfe & approximate Wolfe
function satisfies_wolfe(c::T,
                         phi_c::Real,
                         dphi_c::Real,
                         phi_0::Real,
                         dphi_0::Real,
                         phi_lim::Real,
                         delta::Real,
                         sigma::Real) where T<:Number
    wolfe1 = delta * dphi_0 >= (phi_c - phi_0) / c &&
               dphi_c >= sigma * dphi_0
    wolfe2 = (2 * delta - 1) * dphi_0 >= dphi_c >= sigma * dphi_0 &&
               phi_c <= phi_lim
    return wolfe1 || wolfe2
end

# HZ, stages S1-S4
function ls_secant(a::F64, b::F64, dphi_a::F64, dphi_b::F64)
    return (a * dphi_b - b * dphi_a) / (dphi_b - dphi_a)
end

# phi
function ls_secant2!(ϕdϕ, alphas::Vector{F64},
                  values::Vector{F64}, slopes::Vector{F64},
                  ia::I64, ib::I64,
                  phi_lim::F64, delta::F64, sigma::F64)
    phi_0 = values[1]
    dphi_0 = slopes[1]
    a = alphas[ia]
    b = alphas[ib]
    dphi_a = slopes[ia]
    dphi_b = slopes[ib]
    T = eltype(slopes)
    zeroT = convert(T, 0)
    if !(dphi_a < zeroT && dphi_b >= zeroT)
        error(string("Search direction is not a direction of descent; ",
                     "this error may indicate that user-provided derivatives are inaccurate. ",
                      @sprintf "(dphi_a = %f; dphi_b = %f)" dphi_a dphi_b))
    end
    c = ls_secant(a, b, dphi_a, dphi_b)
    @assert isfinite(c)
    # phi_c = phi(tmpc, c) # Replace
    phi_c, dphi_c = ϕdϕ(c)
    @assert isfinite(phi_c) && isfinite(dphi_c)

    push!(alphas, c)
    push!(values, phi_c)
    push!(slopes, dphi_c)

    ic = length(alphas)
    if satisfies_wolfe(c, phi_c, dphi_c, phi_0, dphi_0, phi_lim, delta, sigma)
        return true, ic, ic
    end

    iA, iB = ls_update!(ϕdϕ, alphas, values, slopes, ia, ib, ic, phi_lim)
    a = alphas[iA]
    b = alphas[iB]
    if iB == ic
        # we updated b, make sure we also update a
        c = ls_secant(alphas[ib], alphas[iB], slopes[ib], slopes[iB])
    elseif iA == ic
        # we updated a, do it for b too
        c = ls_secant(alphas[ia], alphas[iA], slopes[ia], slopes[iA])
    end
    if (iA == ic || iB == ic) && a <= c <= b
        # phi_c = phi(tmpc, c) # TODO: Replace
        phi_c, dphi_c = ϕdϕ(c)
        @assert isfinite(phi_c) && isfinite(dphi_c)

        push!(alphas, c)
        push!(values, phi_c)
        push!(slopes, dphi_c)

        ic = length(alphas)
        # Check arguments here
        if satisfies_wolfe(c, phi_c, dphi_c, phi_0, dphi_0, phi_lim, delta, sigma)
            return true, ic, ic
        end
        iA, iB = ls_update!(ϕdϕ, alphas, values, slopes, iA, iB, ic, phi_lim)
    end
    return false, iA, iB
end

# HZ, stages U0-U3
# Given a third point, pick the best two that retain the bracket
# around the minimum (as defined by HZ, eq. 29)
# b will be the upper bound, and a the lower bound
function ls_update!(ϕdϕ, alphas::Vector{F64},
                    values::Vector{F64}, slopes::Vector{F64},
                    ia::I64, ib::I64, ic::I64, phi_lim::F64)
    a = alphas[ia]
    b = alphas[ib]
    T = eltype(slopes)
    zeroT = convert(T, 0)

    # Debugging (HZ, eq. 4.4):
    @assert slopes[ia] < zeroT
    @assert values[ia] <= phi_lim
    @assert slopes[ib] >= zeroT
    @assert b > a
    c = alphas[ic]
    phi_c = values[ic]
    dphi_c = slopes[ic]
    if c < a || c > b
        return ia, ib #, 0, 0  # it's out of the bracketing interval
    end
    if dphi_c >= zeroT
        return ia, ic #, 0, 0  # replace b with a closer point
    end
    # We know dphi_c < 0. However, phi may not be monotonic between a
    # and c, so check that the value is also smaller than phi_0.  (It's
    # more dangerous to replace a than b, since we're leaving the
    # secure environment of alpha=0; that's why we didn't check this
    # above.)
    if phi_c <= phi_lim
        return ic, ib#, 0, 0  # replace a
    end
    # phi_c is bigger than phi_0, which implies that the minimum
    # lies between a and c. Find it via bisection.
    return ls_bisect!(ϕdϕ, alphas, values, slopes, ia, ic, phi_lim)
end

# HZ, stage U3 (with theta=0.5)
function ls_bisect!(ϕdϕ, alphas::Vector{F64}, values::Vector{F64},
                    slopes::Vector{F64}, ia::I64, ib::I64, phi_lim::F64)
    T = eltype(alphas)
    gphi = convert(T, NaN)
    a = alphas[ia]
    b = alphas[ib]

    # Debugging (HZ, conditions shown following U3)
    zeroT = convert(T, 0)
    @assert slopes[ia] < zeroT
    @assert values[ia] <= phi_lim
    @assert slopes[ib] < zeroT       # otherwise we wouldn't be here
    @assert values[ib] > phi_lim
    @assert b > a
    while b - a > eps(b)
        d = (a + b) / convert(T, 2)
        phi_d, gphi = ϕdϕ(d)
        @assert isfinite(phi_d) && isfinite(gphi)

        push!(alphas, d)
        push!(values, phi_d)
        push!(slopes, gphi)

        id = length(alphas)
        if gphi >= zeroT
            return ia, id # replace b, return
        end
        if phi_d <= phi_lim
            a = d # replace a, but keep bisecting until dphi_b > 0
            ia = id
        else
            b = d
            ib = id
        end
    end
    return ia, ib
end
