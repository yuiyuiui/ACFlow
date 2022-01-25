
function make_data(rd::RawData)
    return make_data(rd.value, rd.error)
end

function make_data(val::Vector{T}, err::Vector{T}) where {T}
    grid = get_c("grid")
    if grid == "matsubara"
        value = vcat(real(val), imag(val))
        error = vcat(real(err), imag(err))
        covar = error .^ 2.0
        _data = GreenData(value, error, covar)
        return _data
    else
        covar = err .^ 2.0
        _data = GreenData(val, err, covar)
        return _data
    end
end

function write_spectrum(am::AbstractMesh, spectra::Vector{F64})
    @assert am.nmesh == length(spectra)
    open("spectra.data", "w") do fout
        for i = 1:am.nmesh
            @printf(fout, "%16.12f %16.12f\n", am[i], spectra[i])
        end
    end
end

function read_time_data(finput::String, ngrid::I64)
end

function read_freq_data(finput::String, ngrid::I64)
    mesh = zeros(F64, ngrid)
    value = zeros(C64, ngrid)
    error = zeros(C64, ngrid)

    open(finput, "r") do fin
        for i = 1:ngrid
            arr = parse.(F64, line_to_array(fin)[1:3])
            mesh[i] = arr[1]
            value[i] = arr[2] + im * arr[3]
            error[i] = 0.0001 + 0.0001im
        end
    end

    return RawData(mesh, value, error)
end