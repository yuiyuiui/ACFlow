#!/usr/bin/env julia

using DelimitedFiles
using Printf

# Number of grid points for input data
ngrid = 100

# Read Matsubara green's function from solver.grn.dat
dlm = readdlm("solver.grn.dat")
grid = dlm[1:ngrid,2]
grn  = dlm[1:ngrid,3] + im * dlm[1:ngrid,4]
err  = dlm[1:ngrid,5] + im * dlm[1:ngrid,6]

# Write green's function
open("green.data", "w") do fout
    for i in eachindex(grid)
        zg = grn[i]
        ze = err[i]
        @printf(fout, "%20.16f %20.16f %20.16f %20.16f %20.16f\n",
            grid[i], real(zg), imag(zg), real(ze), imag(ze))
    end
end

# Read Matsubara self-energy function from solver.sgm.dat
dlm = readdlm("solver.sgm.dat")
grid = dlm[1:ngrid,2]
sgm  = dlm[1:ngrid,3] + im * dlm[1:ngrid,4]
err  = dlm[1:ngrid,5] + im * dlm[1:ngrid,6]

# Write self-energy function
open("sigma.data", "w") do fout
    for i in eachindex(grid)
        zs = sgm[i]
        ze = err[i]
        @printf(fout, "%20.16f %20.16f %20.16f %20.16f %20.16f\n",
            grid[i], real(zs), imag(zs), real(ze), imag(ze))
    end
end
