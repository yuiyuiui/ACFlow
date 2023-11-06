mutable struct InitialStatic{T}
    alpha::T
    scaled::Bool
end

function InitialStatic()
    InitialStatic(1.0, false)
end

function (is::InitialStatic{T})(state::BFGSState) where T
    PT = promote_type(T, real(eltype(state.ls)))
    if is.scaled == true && (ns = real(norm(state.ls))) > convert(PT, 0)
        # TODO: Type instability if there's a type mismatch between is.alpha and ns?
        state.alpha = convert(PT, min(is.alpha, ns)) / ns
    else
        state.alpha = convert(PT, is.alpha)
    end
end

mutable struct LineSearchException{T<:Real} <: Exception
    message::AbstractString
    alpha::T
end

function make_ϕ(df::BFGSDifferentiable, x_new, x, s)
    function ϕ(α)
        # Move a distance of alpha in the direction of s
        x_new .= x .+ α.*s

        # Evaluate f(x+α*s)
        value!(df, x_new)
    end
    ϕ
end

function make_ϕ_ϕdϕ(df::BFGSDifferentiable, x_new, x, s)
    function ϕdϕ(α)
        # Move a distance of alpha in the direction of s
        x_new .= x .+ α.*s

        # Evaluate ∇f(x+α*s)
        value_gradient!(df, x_new)

        # Calculate ϕ'(a_i)
        value(df), real(dot(gradient(df), s))
    end
    make_ϕ(df, x_new, x, s), ϕdϕ
end

#
# Conjugate gradient line search implementation from:
#   W. W. Hager and H. Zhang (2006) Algorithm 851: CG_DESCENT, a
#     conjugate gradient method with guaranteed descent. ACM
#     Transactions on Mathematical Software 32: 113–137.
#

# Display flags are represented as a bitfield
# (not exported, but can use via LineSearches.ITER, for example)
const one64 = convert(UInt64, 1)
const BRACKET     = one64 << 8
const LINESEARCH  = one64 << 9
const UPDATE      = one64 << 10
const SECANT2     = one64 << 11
const BISECT      = one64 << 12
const DEFAULTDELTA = 0.1 # Values taken from HZ paper (Nocedal & Wright recommends 0.01?)
const DEFAULTSIGMA = 0.9 # Values taken from HZ paper (Nocedal & Wright recommends 0.1 for GradientDescent)

"""
Conjugate gradient line search implementation from:
  W. W. Hager and H. Zhang (2006) Algorithm 851: CG_DESCENT, a
    conjugate gradient method with guaranteed descent. ACM
    Transactions on Mathematical Software 32: 113–137.
"""

mutable struct HagerZhang{T, Tm}
    delta::T # DEFAULTDELTA # c_1 Wolfe sufficient decrease condition
    sigma::T # DEFAULTSIGMA # c_2 Wolfe curvature condition (Recommend 0.1 for GradientDescent)
    alphamax::T # Inf
    rho::T # 5.0
    epsilon::T # 1e-6
    gamma::T # 0.66
    linesearchmax::Int # 50
    psi3::T # 0.1
    display::Int # 0
    mayterminate::Tm # Ref{Bool}(false)
 end

function HagerZhang()
    HagerZhang(
   DEFAULTDELTA, # c_1 Wolfe sufficient decrease condition
   DEFAULTSIGMA, # c_2 Wolfe curvature condition (Recommend 0.1 for GradientDescent)
   Inf,
   5.0,
   1e-6,
   0.66,
   50,
   0.1,
   0,
   Ref{Bool}(false)
   )
end

HagerZhang{T}(args...; kwargs...) where T = HagerZhang{T, Base.RefValue{Bool}}(args...; kwargs...)

function (ls::HagerZhang)(df::BFGSDifferentiable, x::AbstractArray{T},
                          s::AbstractArray{T}, α::Real,
                          #x_new::AbstractArray{T},
                          phi_0::Real, dphi_0::Real) where T
    x_new = similar(x)
    ϕ, ϕdϕ = make_ϕ_ϕdϕ(df, x_new, x, s)
    ls(ϕ, ϕdϕ, α::Real, phi_0, dphi_0)
end

(ls::HagerZhang)(ϕ, dϕ, ϕdϕ, c, phi_0, dphi_0) = ls(ϕ, ϕdϕ, c, phi_0, dphi_0)

#function unpack end
@inline unpack(x, ::Val{f}) where {f} = getproperty(x, f)
@inline unpack(x::AbstractDict{Symbol}, ::Val{k}) where {k} = x[k]
@inline unpack(x::AbstractDict{<:AbstractString}, ::Val{k}) where {k} = x[string(k)]

macro unpack(args)
    args.head!=:(=) && error("Expression needs to be of form `a, b = c`")
    items, suitecase = args.args
    items = isa(items, Symbol) ? [items] : items.args
    suitecase_instance = gensym()
    kd = [:( $key = unpack($suitecase_instance, Val{$(Expr(:quote, key))}()) ) for key in items]
    kdblock = Expr(:block, kd...)
    expr = quote
        local $suitecase_instance = $suitecase # handles if suitecase is not a variable but an expression
        $kdblock
        $suitecase_instance # return RHS of `=` as standard in Julia
    end
    esc(expr)
end

# TODO: Should we deprecate the interface that only uses the ϕ and ϕd\phi arguments?
function (ls::HagerZhang)(ϕ, ϕdϕ,
                          c::T,
                          phi_0::Real,
                          dphi_0::Real) where T # Should c and phi_0 be same type?
    @unpack delta, sigma, alphamax, rho, epsilon, gamma,
            linesearchmax, psi3, display, mayterminate = ls

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
    if display & LINESEARCH > 0
        println("New linesearch")
    end

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
        if display & LINESEARCH > 0
            println("Wolfe condition satisfied on point alpha = ", c)
        end
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
        if display & BRACKET > 0
            println("bracketing: ia = ", ia,
                    ", ib = ", ib,
                    ", c = ", c,
                    ", phi_c = ", phi_c,
                    ", dphi_c = ", dphi_c)
        end
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
            ia, ib = bisect!(ϕdϕ, alphas, values, slopes, ia, ib, phi_lim, display)
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
                if display & BRACKET > 0
                    println("bracket: exceeding alphamax, using c = alphamax = ", alphamax,
                    ", cold = ", cold)
                end
            end
            phi_c, dphi_c = ϕdϕ(c)
            iterfinite = 1
            while !(isfinite(phi_c) && isfinite(dphi_c)) && c > nextfloat(cold) && iterfinite < iterfinitemax
                alphamax = c # shrinks alphamax, assumes that steps >= c can never have finite phi_c and dphi_c
                iterfinite += 1
                if display & BRACKET > 0
                    println("bracket: non-finite value, bisection")
                end
                c = (cold + c) / 2
                phi_c, dphi_c = ϕdϕ(c)
            end
            if !(isfinite(phi_c) && isfinite(dphi_c))
                if display & BRACKET > 0
                    println("Warning: failed to expand interval to bracket with finite values. If this happens frequently, check your function and gradient.")
                    println("c = ", c,
                            ", alphamax = ", alphamax,
                            ", phi_c = ", phi_c,
                            ", dphi_c = ", dphi_c)
                end
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
        if display & LINESEARCH > 0
            println("linesearch: ia = ", ia,
                    ", ib = ", ib,
                    ", a = ", a,
                    ", b = ", b,
                    ", phi(a) = ", values[ia],
                    ", phi(b) = ", values[ib])
        end
        if b - a <= eps(b)
            mayterminate[] = false # reset in case another initial guess is used next
            return a, values[ia] # lsr.value[ia]
        end
        iswolfe, iA, iB = secant2!(ϕdϕ, alphas, values, slopes, ia, ib, phi_lim, delta, sigma, display)
        if iswolfe
            mayterminate[] = false # reset in case another initial guess is used next
            return alphas[iA], values[iA] # lsr.value[iA]
        end
        A = alphas[iA]
        B = alphas[iB]
        @assert B > A
        if B - A < gamma * (b - a)
            if display & LINESEARCH > 0
                println("Linesearch: secant succeeded")
            end
            if nextfloat(values[ia]) >= values[ib] && nextfloat(values[iA]) >= values[iB]
                # It's so flat, secant didn't do anything useful, time to quit
                if display & LINESEARCH > 0
                    println("Linesearch: secant suggests it's flat")
                end
                mayterminate[] = false # reset in case another initial guess is used next
                return A, values[iA]
            end
            ia = iA
            ib = iB
        else
            # Secant is converging too slowly, use bisection
            if display & LINESEARCH > 0
                println("Linesearch: secant failed, using bisection")
            end
            c = (A + B) / convert(T, 2)

            phi_c, dphi_c = ϕdϕ(c)
            @assert isfinite(phi_c) && isfinite(dphi_c)
            push!(alphas, c)
            push!(values, phi_c)
            push!(slopes, dphi_c)

            ia, ib = update!(ϕdϕ, alphas, values, slopes, iA, iB, length(alphas), phi_lim, display)
        end
        iter += 1
    end

    throw(LineSearchException("Linesearch failed to converge, reached maximum iterations $(linesearchmax).",
                              alphas[ia]))
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
function secant(a::Real, b::Real, dphi_a::Real, dphi_b::Real)
    return (a * dphi_b - b * dphi_a) / (dphi_b - dphi_a)
end

function secant(alphas, values, slopes, ia::Integer, ib::Integer)
    return secant(alphas[ia], alphas[ib], slopes[ia], slopes[ib])
end

# phi
function secant2!(ϕdϕ,
                  alphas,
                  values,
                  slopes,
                  ia::Integer,
                  ib::Integer,
                  phi_lim::Real,
                  delta::Real = DEFAULTDELTA,
                  sigma::Real = DEFAULTSIGMA,
                  display::Integer = 0)
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
    c = secant(a, b, dphi_a, dphi_b)
    if display & SECANT2 > 0
        println("secant2: a = ", a, ", b = ", b, ", c = ", c)
    end
    @assert isfinite(c)
    # phi_c = phi(tmpc, c) # Replace
    phi_c, dphi_c = ϕdϕ(c)
    @assert isfinite(phi_c) && isfinite(dphi_c)

    push!(alphas, c)
    push!(values, phi_c)
    push!(slopes, dphi_c)

    ic = length(alphas)
    if satisfies_wolfe(c, phi_c, dphi_c, phi_0, dphi_0, phi_lim, delta, sigma)
        if display & SECANT2 > 0
            println("secant2: first c satisfied Wolfe conditions")
        end
        return true, ic, ic
    end

    iA, iB = update!(ϕdϕ, alphas, values, slopes, ia, ib, ic, phi_lim, display)
    if display & SECANT2 > 0
        println("secant2: iA = ", iA, ", iB = ", iB, ", ic = ", ic)
    end
    a = alphas[iA]
    b = alphas[iB]
    doupdate = false
    if iB == ic
        # we updated b, make sure we also update a
        c = secant(alphas, values, slopes, ib, iB)
    elseif iA == ic
        # we updated a, do it for b too
        c = secant(alphas, values, slopes, ia, iA)
    end
    if (iA == ic || iB == ic) && a <= c <= b
        if display & SECANT2 > 0
            println("secant2: second c = ", c)
        end
        # phi_c = phi(tmpc, c) # TODO: Replace
        phi_c, dphi_c = ϕdϕ(c)
        @assert isfinite(phi_c) && isfinite(dphi_c)

        push!(alphas, c)
        push!(values, phi_c)
        push!(slopes, dphi_c)

        ic = length(alphas)
        # Check arguments here
        if satisfies_wolfe(c, phi_c, dphi_c, phi_0, dphi_0, phi_lim, delta, sigma)
            if display & SECANT2 > 0
                println("secant2: second c satisfied Wolfe conditions")
            end
            return true, ic, ic
        end
        iA, iB = update!(ϕdϕ, alphas, values, slopes, iA, iB, ic, phi_lim, display)
    end
    if display & SECANT2 > 0
        println("secant2 output: a = ", alphas[iA], ", b = ", alphas[iB])
    end
    return false, iA, iB
end

# HZ, stages U0-U3
# Given a third point, pick the best two that retain the bracket
# around the minimum (as defined by HZ, eq. 29)
# b will be the upper bound, and a the lower bound
function update!(ϕdϕ,
                 alphas,
                 values,
                 slopes,
                 ia::Integer,
                 ib::Integer,
                 ic::Integer,
                 phi_lim::Real,
                 display::Integer = 0)
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
    if display & UPDATE > 0
        println("update: ia = ", ia,
                ", a = ", a,
                ", ib = ", ib,
                ", b = ", b,
                ", c = ", c,
                ", phi_c = ", phi_c,
                ", dphi_c = ", dphi_c)
    end
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
    return bisect!(ϕdϕ, alphas, values, slopes, ia, ic, phi_lim, display)
end

# HZ, stage U3 (with theta=0.5)
function bisect!(ϕdϕ,
                 alphas::AbstractArray{T},
                 values,
                 slopes,
                 ia::Integer,
                 ib::Integer,
                 phi_lim::Real,
                 display::Integer = 0) where T
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
        if display & BISECT > 0
            println("bisect: a = ", a, ", b = ", b, ", b - a = ", b - a)
        end
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
