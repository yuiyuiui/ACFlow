push!(LOAD_PATH, ENV["ACFLOW_HOME"])

using Documenter, ACFlow, Random

makedocs(
    sitename = "ACFlow",
    clean = false,
    authors = "Li Huang",
    format = Documenter.HTML(
        prettyurls = false,
        ansicolor = true,
    ),
    modules = [ACFlow],
    pages = [
        "Home" => "index.md",
        "Introduction" => "intro.md",
        "Manual" => Any[
            "Features" => "man/feature.md",
            "Installation" => "man/install.md",
            "Input" => "man/input.md",
            "Output" => "man/output.md",
            "Parameters" => "man/param.md",
        ],
        "Examples" => Any[
            "T01" => "examples/T01.md",
            "T02" => "examples/T02.md",
            "T03" => "examples/T03.md",
        ],
        "Theory" => Any[
            "Maximum Entropy Method" => "theory/maxent.md",
            "Stochastic Analytical Continuation" => "theory/sac.md",
            "Stochastic Optimization Method" => "theory/som.md",
        ],
        "Library" => Any[
            "Outline" => "library/outline.md",
            "ACFlow" => "library/acflow.md",
            "Constants" => "library/global.md",
            "Types" => "library/types.md",
            "Core" => "library/base.md",
            "Solvers" => "library/solver.md",
            "Grid" => "library/grid.md",
            "Mesh" => "library/mesh.md",
            "Models" => "library/model.md",
            "Kernels" => "library/kernel.md",
            "Configuration" => "library/config.md",
            "Input and output" => "library/inout.md",
            "Math" => "library/math.md",
            "Utilities" => "library/util.md",
        ],
    ],
)
