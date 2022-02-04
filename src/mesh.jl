#
# Project : Gardenia
# Source  : mesh.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2022/02/04
#

abstract type AbstractMesh end

mutable struct LinearMesh <: AbstractMesh
    nmesh :: I64
    wmax :: F64
    wmin :: F64
    mesh :: Vector{F64}
    weight :: Vector{F64}
end

function LinearMesh(nmesh::I64, wmin::F64, wmax::F64)
    @assert nmesh ≥ 1
    @assert wmax > wmin

    mesh = collect(LinRange(wmin, wmax, nmesh))
    weight = (mesh[2:end] + mesh[1:end-1]) / 2.0
    pushfirst!(weight, mesh[1])
    push!(weight, mesh[end])
    weight = diff(weight)

    return LinearMesh(nmesh, wmax, wmin, mesh, weight)
end

function Base.eachindex(lm::LinearMesh)
    eachindex(um.mesh)
end

function Base.firstindex(lm::LinearMesh)
    firstindex(um.mesh)
end

function Base.lastindex(lm::LinearMesh)
    lastindex(um.mesh)
end

function Base.getindex(lm::LinearMesh, ind::I64)
    @assert 1 ≤ ind ≤ um.nmesh
    return um.mesh[ind]
end

function Base.getindex(lm::LinearMesh, I::UnitRange{I64})
    @assert checkbounds(Bool, um.mesh, I)
    lI = length(I)
    X = similar(um.mesh, lI)
    if lI > 0
        unsafe_copyto!(X, 1, um.mesh, first(I), lI)
    end
    return X
end

mutable struct TangentMesh <: AbstractMesh
    nmesh :: I64
    wmax :: F64
    wmin :: F64
    mesh :: Vector{F64}
    weight :: Vector{F64}
end

function TangentMesh(nmesh::I64, wmin::F64, wmax::F64)
    @assert nmesh ≥ 1
    @assert wmax > 0.0 > wmin
    @assert wmax == abs(wmin)

    f1 = 2.1
    mesh = collect(LinRange(-π / f1, π / f1, nmesh))
    mesh = wmax * tan.(mesh) / tan(π / f1)
    weight = (mesh[2:end] + mesh[1:end-1]) / 2.0
    pushfirst!(weight, mesh[1])
    push!(weight, mesh[end])
    weight = diff(weight)

    return TangentMesh(nmesh, wmax, wmin, mesh, weight)
end

function Base.eachindex(tm::TangentMesh)
    eachindex(tm.mesh)
end

function Base.firstindex(tm::TangentMesh)
    firstindex(tm.mesh)
end

function Base.lastindex(tm::TangentMesh)
    lastindex(tm.mesh)
end

function Base.getindex(tm::TangentMesh, ind::I64)
    @assert 1 ≤ ind ≤ tm.nmesh
    return tm.mesh[ind]
end

function Base.getindex(tm::TangentMesh, I::UnitRange{I64})
    @assert checkbounds(Bool, tm.mesh, I)
    lI = length(I)
    X = similar(tm.mesh, lI)
    if lI > 0
        unsafe_copyto!(X, 1, tm.mesh, first(I), lI)
    end
    return X
end

function trapz(x::Vector{F64}, y::Vector{T}, uniform::Bool = false) where {T}
    if uniform
        h = x[2] - x[1]
        _sum = sum(y[2:end-1])
        value = (h / 2.0) * (y[1] + y[end] + 2.0 * _sum)
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

function trapz(x::AbstractMesh, y::Vector{T}) where {T}
    value = dot(x.weight, y)
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