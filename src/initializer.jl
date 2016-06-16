abstract Initializer

type NormalInit <: Initializer
    mu :: AbstractFloat
    sd :: AbstractFloat
    function NormalInit(mu=0.0, sd=1.0)
        new(mu, sd)
    end
end

type UniformInit <: Initializer
    min :: AbstractFloat
    max :: AbstractFloat
    function UniformInit(min=0.0, max=1.0)
        new(min, max)
    end
end

function initialize(T::DataType, mapdim::Tuple, init::UniformInit)
    data = rand(T, mapdim)
    data -= T(init.min)
    data /= T(init.max-init.min)
end

function initialize(T::DataType, mapdim::Tuple, init::NormalInit)
    data = convert(Array{T},randn(mapdim))
    data *= T(init.sd)
    data += T(init.mu)
end

