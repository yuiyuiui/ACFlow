# Running Modes

The ACFlow toolkit is designed to be flexible and easy-to-use. It provides three running modes to facilitate analytical continuation calculations, namely the interactive, script, and standard modes.  

## Interactive Mode

With the ACFlow toolkit, the users can setup and carry out analytical continuation simulations interactively in Julia's REPL (Read-Eval-Print Loop) environment. For example,

>    julia> push!(LOAD_PATH, ENV["ACFLOW_HOME"])
>
>    julia> using ACFlow
>
>    julia> setup_args("ac.toml")
>
>    julia> read_param()
>
>    julia> mesh, Aout, Gout = solve(read_data())

Here, `ac.toml` is a configuration file, which contains essential computational parameters. The return values of the `solve()` function (i.e., `mesh`, `Aout`, and `Gout`) are mesh at real axis ``\omega``, spectral density ``A(\omega)``, and reproduced Green's function ``\tilde{G}``, respectively. They can be further analyzed or visualized by the users.  

## Script Mode

The core functionalities of the ACFlow toolkit are exposed to the users via a simple application programming interface. So, the users can write Julia scripts easily by themselves to perform analytical continuation simulations. A minimal Julia script (`acrun.jl`) is listed as follows:

>    #!/usr/bin/env julia
>
>    push!(LOAD_PATH, ENV["ACFLOW_HOME"])
>
>    using ACFlow
>
>    setup_args("ac.toml")
>
>    read_param()
>
>    mesh, Aout, Gout = solve(read_data())

Of course, this script can be extended to finish complex tasks. In section~\ref{subsec:sigma}, a realistic example is provided to show how to complete an analytical continuation of Matsubara self-energy function via the script mode.              

## Standard Mode

In the standard mode, the users have to prepare the input data manually. In addition, a configuration file must be provided. Supposed that the configuration file is `ac.toml`, then the analytical continuation calculation is launched as follows:

> $ /home/your_home/acflow/util/acrun.jl ac.toml

or

> $ /home/your_home/acflow/util/Pacrun.jl ac.toml

Noted that the `acrun.jl` script runs sequentially, while the `Pacrun.jl` script supports parallel and distributed computing. As we can conclude from the filename extension of configuration file (`ac.toml`), it adopts the `TOML` specification. The users may edit it with any text-based editors. Next we will introduce syntax and format of the input data files and configuration files.