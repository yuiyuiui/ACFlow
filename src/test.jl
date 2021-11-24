push!(LOAD_PATH, ".")

using ACFlow

println("Hello world! This is a maxent code.")
ω = FermionicMatsubaraGrid()
𝐺 = GreenData()
read_data!(ω, 𝐺)
𝑀 = calc_moments(ω, 𝐺)
@show 𝑀