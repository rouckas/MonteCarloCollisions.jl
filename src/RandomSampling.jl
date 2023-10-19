using StaticArrays
using Random: rand, randn, randexp

@inline function random_direction(l=1.)
    """generate a 3D vector with random direction and length l"""
    cos_theta =  1. - 2*rand()
    sin_theta = sqrt(1. - cos_theta^2)
    sin_cos_phi = sincos(2*pi*rand())
    SVector(
        l*cos_theta,
        l*sin_theta * sin_cos_phi[1],
        l*sin_theta * sin_cos_phi[2]
    )
end

@inline function random_maxwell_v(vthermal)
    SVector(randn(),randn(), randn())*vthermal # sqrt(k_B*T/m)
end

@inline function random_maxwell_vflux(vthermal)
    sqrt(randn()^2 + randn()^2)*vthermal # sqrt(k_B*T/m)
end

@inline function choose(probs)
    # equivalent to dist=Categorical(probs); rand(dist),
    # > 30% faster, but no error checking (sum(probs) == 1 ?)
    l = length(probs)
    i = 1
    if l > 1
        U = rand()
        @inbounds while (U -= probs[i]) > 0 && i < l
            i += 1
        end
    end
    i
end