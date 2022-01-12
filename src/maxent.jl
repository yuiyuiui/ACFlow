#
# Project : Gardenia
# Source  : maxent.jl
# Author  : Li Huang (lihuang.dmft@gmail.com)
# Status  : Unstable
#
# Last modified: 2022/01/12
#

using Statistics
using LinearAlgebra

@inline function line_to_array(io::IOStream)
    split(readline(io), " ", keepempty = false)
end

const I64 = Int64
const F64 = Float64
const C64 = ComplexF64
const N64 = Union{I64,F64,C64}

abstract type AbstractData end
abstract type AbstractGrid end

struct GreenData <: AbstractData
    value :: Vector{N64}
    error :: Vector{N64}
    covar :: Vector{N64}
end

struct ImaginaryTimeGrid <: AbstractGrid
    grid :: Vector{F64}
end

struct FermionicMatsubaraGrid <: AbstractGrid
    grid :: Vector{F64}
end

function read_data!(::Type{FermionicMatsubaraGrid})
    grid  = F64[] 
    value = C64[]
    error = C64[]
    covar = F64[]

    niw = 10
    #
    open("green.data", "r") do fin
        for i = 1:niw
            arr = parse.(F64, line_to_array(fin))
            push!(grid, arr[1])
            push!(value, arr[2] + arr[3] * im)
        end
    end
    #=
    open("err.data", "r") do fin
        for i = 1:niw
            arr = parse.(F64, line_to_array(fin))
            @assert grid[i] == arr[1]
            push!(error, arr[2] + arr[3] * im)
            push!(covar, arr[2]^2)
            push!(covar, arr[3]^2)
        end
    end
    =#

    return FermionicMatsubaraGrid(grid), GreenData(value, error, covar)
end

println("hello")
ω, G = read_data!(FermionicMatsubaraGrid)