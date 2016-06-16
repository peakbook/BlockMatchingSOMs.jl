isdefined(Base, :__precompile__) && __precompile__()

module BlockMatchingSOMs

using Distances

include("initializer.jl")
include("bmsom.jl")

end
