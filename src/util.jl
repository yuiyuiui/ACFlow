#
# Project : Gardenia
# Source  : util.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2022/02/05
#

#=
### *Basic Macros*
=#

"""
    @cswitch(constexpr, body)

Provides a C-like switch statement with the *falling through* behavior.
This implementation was borrowed from the following github repository:

* https://github.com/Gnimuc/CSyntax.jl

### Examples
```julia
engine = get_d("engine")
@cswitch engine begin
    @case "vasp"
        just_do_it()
        break

    @default
        sorry()
        break
end
```
"""
macro cswitch(constexpr, body)
    case2label = Dict{Any,Symbol}()
    flow = Expr(:block)
    end_label = gensym("end")
    default_label = end_label

    for arg in body.args
        if Meta.isexpr(arg, :macrocall) && arg.args[1] == Symbol("@case")
            label = gensym("case")
            case2label[arg.args[3]] = label
            labelexpr = Expr(:symboliclabel, label)
            push!(flow.args, labelexpr)
        elseif Meta.isexpr(arg, :macrocall) && arg.args[1] == Symbol("@default")
            default_label = gensym("default")
            labelexpr = Expr(:symboliclabel, default_label)
            push!(flow.args, labelexpr)
        elseif arg == Expr(:break)
            labelexpr = Expr(:symbolicgoto, end_label)
            push!(flow.args, labelexpr)
        else
            push!(flow.args, arg)
        end
    end
    push!(flow.args, Expr(:symboliclabel, end_label))

    jumptable = Expr(:block)
    for (case, label) in case2label
        condition = Expr(:call, :(==), constexpr, case)
        push!(jumptable.args, Expr(:if, condition, Expr(:symbolicgoto, label)))
    end
    push!(jumptable.args[end].args, Expr(:symbolicgoto, default_label))

    return esc(Expr(:block, jumptable, flow))
end

"""
    @time_call(ex)

Evaluate a function call (`ex`), and then print the elapsed time (number
of seconds) it took to execute.

This macro is a variation of the standard `@elapsed` macro.
"""
macro time_call(ex)
    quote
        while false; end
        local t₀ = time_ns()
        $(esc(ex))
        δt = (time_ns() - t₀) / 1e9
        println("Report: Total elapsed time $(δt) s\n")
        flush(stdout)
    end
end

#=
### *Query Runtime Environment*
=#

"""
    require()

Check the version of julia runtime environment. It should be higher
than v1.6.x. One of the most important philosophies of the `ACFlow`
package is minimizing the dependence on the third-party libraries as
far as possible. Note that the `ACFlow` package relys on the `TOML`
package to parse the *.toml file. Only in v1.6.0 and higher versions,
julia includes the `TOML` package in its standard library.
"""
function require()
    if VERSION < v"1.6-"
        error("Please upgrade your julia to v1.6.0 or higher")
    end
end

@inline function line_to_array(io::IOStream)
    split(readline(io), " ", keepempty = false)
end

"""
    sorry()

Print an error message to the screen.
"""
function sorry()
    error("Sorry, this feature has not been implemented")
end

"""
    query_args()

Check whether the configuration file (`case.toml`) is provided.

See also: [`setup_args`](@ref).
"""
function query_args()
    nargs = length(ARGS)
    if nargs < 1
        error("Please specify the configuration file")
    else
        ARGS[1]
    end
end

#function myfun(a, b)
#    #return a ^ 2.0 + b * a + 1.0
#    return a ^ 3.0 - b
#end

#s = secant(myfun, 1.0, 8.0)
#println(s)

function secant(func, x0, args)
    eps = 1.0e-4
    maxiter = 50
    tol = 1.48e-8
    funcalls = 0
    p0 = 1.0 * x0
    p1 = x0 * (1.0 + eps)
    if p1 ≥ 0.0
        p1 = p1 + eps
    else
        p1 = p1 - eps
    end

    q0 = func(p0, args)
    funcalls = funcalls + 1
    q1 = func(p1, args)
    funcalls = funcalls + 1

    if abs(q1) < abs(q0)
        p0, p1 = p1, p0
        q0, q1 = q1, q0
    end

    for itr = 1:maxiter
        if q1 == q0
            if p1 != p0
                error("tolerance is reached!")
            end
            p = (p1 + p0) / 2.0
            return p
        else
            if abs(q1) > abs(q0)
                p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
            else
                p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)
            end
        end

        if abs(p - p1) < tol
            return p
        end

        p0, q0 = p1, q1
        p1 = p
        q1 = func(p1, args)
        funcalls = funcalls + 1
    end
end

function newton(fun::Function, guess, kwargs...)
    max_iter = 20000
    mixing = 0.5
    counter = 0
    result = nothing

    function apply(prop::Vector{T}, f::Vector{T}, J::Matrix{T}) where {T}
        resid = nothing
        step = 1.0
        limit = 1e-4
    
        try
            resid = - pinv(J) * f
        catch
            resid = zeros(F64, length(prop))
        end
    
        if any(x -> x > limit, abs.(prop))
            ratio = abs.(resid ./ prop)
            max_ratio = maximum( ratio[ abs.(prop) .> limit ] )
            if max_ratio > 1.0
                step = 1.0 / max_ratio
            end
        end
    
        result = prop + step .* resid
    
        return result
    end

    props = []
    reals = []

    f, J = fun(guess, kwargs...)
    init = apply(guess, f, J)
    push!(props, guess)
    push!(reals, init)

    while true
        counter = counter + 1

        prop = props[end] + mixing * (reals[end] - props[end])
        f, J = fun(prop, kwargs...)
        result = apply(prop, f, J)

        push!(props, prop)
        push!(reals, result)
        
        any(isnan.(result)) && error("Got NaN!")

        if counter > max_iter || maximum( abs.(result - prop) ) < 1.e-4
            break
        end
    end

    counter > max_iter && error("The max_iter is too small!")

    return result, counter
end

function trapz(x::Vector{F64}, y::Vector{T}, linear::Bool = false) where {T}
    if linear
        h = x[2] - x[1]
        value = y[1] + y[end] + 2.0 * sum(y[2:end-1])
        value = h * value / 2.0
    else
        len = length(x)
        value = 0.0
        for i = 1:len-1
            value = value + (y[i] + y[i+1]) * (x[i+1] - x[i])
        end
        value = value / 2.0    
    end

    return value
end

function simpson(x::Vector{F64}, y::Vector{T}) where {T}
    h = x[2] - x[1]
    even_sum = 0.0
    odd_sum = 0.0

    for i = 1:length(x)-1
        if iseven(i)
            even_sum = even_sum + y[i]
        else
            odd_sum = odd_sum + y[i]
        end
    end

    val = h / 3.0 * (y[1] + y[end] + 2.0 * even_sum + 4.0 * odd_sum)

    return val
end