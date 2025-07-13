# ---------------------------------------------------
# smooth situation
# ---------------------------------------------------

using ACFlow
using Plots
include("../method.jl")

wn, GFV, A = sac_dfcfg_cont()

res = solve(wn, GFV)

plot(res[1], res[2], label="reconstructed A(ω)", xlabel="ω", ylabel="A(ω)", title="StochAC, cont type")
plot!(res[1], A.(res[1]), label="original A(ω)")

# ---------------------------------------------------
# delta situation
# ---------------------------------------------------

using ACFlow
using Plots
include("../method.jl")

wn, GFV, (poles, γ) = sac_dfcfg_delta()

res = solve(wn, GFV)

plot(res[1], res[2], label="reconstructed A(ω)", xlabel="ω", ylabel="A(ω)", title="StochAC, delta type")
scatter!(poles, fill(0.0, length(poles)), 
    label="original poles", 
    markershape=:circle,
    markersize=4,
    markercolor=:red
)