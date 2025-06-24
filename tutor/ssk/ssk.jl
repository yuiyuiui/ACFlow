using ACFlow,Plots
include("/Users/syyui/projects/ACFlow/tutor/method.jl")


wn, GFV, (p,γ) = dfcfg_delta()

res = solve(wn, GFV)

plot(res[1],res[2])

function find_peaks(v,minipeak)
    idx = findall(v,x->x>minipeak)
    diffv