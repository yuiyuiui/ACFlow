# Implementations

The ACFlow toolkit is developed with pure Julia language. Thanks to powerful type system and multiple dispatch paradigm of the Julia language, the four different analytical continuation solvers are integrated into an united software architecture. Redundant codes are greatly reduced. It is quite easy to implement new analytical continuation solver or add new features to the existing solvers in the future. Distributed computing is a built-in feature of Julia. So, it is straightforward to realize parallel calculations in the ACFlow toolkit. Now except for the `MaxEnt` solver, all the other solvers are paralleled.

| Filename | Description |
| :------- | :---------- |
| `ACFlow.jl` | Entry of the ACFlow module. |
| `maxent.jl` | Maxent entropy method. |
| `sac.jl`    | Stochastic analytical continuation (K. S. D. Beach's version). |
| `san.jl`    | Stochastic analytical continuation (A. W. Sandvik's version). |
| `som.jl`    | Stochastic optimization method. |
| `global.jl` | Numerical and physical constants. |
| `types.jl`  | Basic data structures and computational parameters. |
| `base.jl`   | Driver for analytical continuation simulation. |
| `inout.jl`  | Read input data and write calculated results. |
| `config.jl` | Parse configuration file and extract computational parameters. |
| `math.jl`   | Root finding, numerical integration, interpolation, Einstein summation, and curve fitting. |
| `util.jl`   | Some utility functions. |
| `mesh.jl`   | Meshes for spectral density. |
| `grid.jl`   | Grids for input data. |
| `model.jl`  | Default model functions. |
| `kernel.jl` | Kernel functions. |

**Table** List of source codes of the ACFlow toolkit.

The source codes of the ACFlow toolkit are placed in the `acflow/src` folder. Their functions are summarized in Table~\ref{tab:source}. The documentation of the ACFlow toolkit is developed by using the Markdown language and `Documenter.jl` package. The source codes are placed in the `acflow/docs` folder. The users can build documentation by themselves. Please see Section~\ref{sec:usage} for how to do that. Or they can read the latest documentation in the following website:

>    http://huangli712.github.io
  
Ten tests and four tutorials are also shipped with the ACFlow toolkit. Their source codes are placed in the `acflow/test` folder. See `acflow/test/test.md` and `acflow/test/tutor.md` for more details. The code repository of the ACFlow toolkit is:

>    https://github.com/huangli712/ACFlow