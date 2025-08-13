# Cont type

using ACFlow
using Plots
include("../method.jl")

wn, GFV, A = maxent_dfcfg_cont(; opb = 8.0, stype = "br")

res = solve(wn, GFV)

plot(res[1], res[2])
plot!(res[1], A.(res[1]))


# Delta type

using ACFlow
using Plots
include("../method.jl")

wn, GFV, (poles, γ) = maxent_dfcfg_delta(; method = "chi2kink", stype = "br")

res = solve(wn, GFV)

plot(
    res[1],
    res[2],
    label = "reconstructed A(ω)",
    xlabel = "ω",
    ylabel = "A(ω)",
    title = "MaxEnt, delta type",
)
scatter!(
    poles,
    fill(0.0, length(poles)),
    label = "original poles",
    markershape = :circle,
    markersize = 4,
    markercolor = :red,
)
