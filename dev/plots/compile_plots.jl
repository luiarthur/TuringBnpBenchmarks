# See: ttps://julialang.github.io/PackageCompiler.jl/dev/examples/plots/

import Pkg; Pkg.activate(".")
using PackageCompiler

create_sysimage(:Plots,
                sysimage_path="sys_plots.so",
                precompile_execution_file="test.jl")
