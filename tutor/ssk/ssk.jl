using ACFlow
using Plots
include("../method.jl")

wn, GFV, (p,γ) = ssk_dfcfg_delta()

res = solve(wn, GFV)

# plot(res[1],res[2])

rep = res[1][find_peaks(res[2], 1)]

norm(rep- p)