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

using ACFlow
using Plots
include("../method.jl")

wn, GFV, (poles, γ) = spx_dfcfg_delta()

res = solve(wn, GFV)

plot(res[1], res[2], label="reconstructed A(ω)", xlabel="ω", ylabel="A(ω)", title="StochPX, delta type")
scatter!(poles, fill(0.0, length(poles)), 
    label="original poles", 
    markershape=:circle,
    markersize=4,
    markercolor=:red
)