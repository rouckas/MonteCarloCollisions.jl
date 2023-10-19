using StaticArrays
abstract type AbstractParticle end
abstract type AbstractEnsemble end


mutable struct Particle3d3v <: AbstractParticle
    r::SVector{3, Float64}
    v::SVector{3, Float64}
    t::Float64
    tau::Float64
end
Base.zero(::Union{Type{Particle3d3v}, Particle3d3v}) = Particle3d3v(zero(SVector{3, Float64}), zero(SVector{3, Float64}), 0., 0.)


mutable struct Particle1d3v <: AbstractParticle
    r::SVector{1, Float64}
    v::SVector{3, Float64}
    t::Float64
    tau::Float64
end

mutable struct Particle1d3vE <: AbstractParticle
    r::SVector{1, Float64}
    v::SVector{3, Float64}
    E::SVector{3, Float64}
    t::Float64
    tau::Float64
end

Base.zero(::Union{Type{Particle1d3vE}, Particle1d3vE}) = Particle1d3vE(zero(SVector{1, Float64}), zero(SVector{3, Float64}), zero(SVector{3, Float64}), 0., 0.)

mutable struct Particle1d1vE <: AbstractParticle
    r::SVector{1, Float64}
    v::SVector{1, Float64}
    E::SVector{1, Float64}
    t::Float64
    tau::Float64
end

Base.zero(::Union{Type{Particle1d1vE}, Particle1d1vE}) = Particle1d1vE(zero(SVector{1, Float64}), zero(SVector{1, Float64}), zero(SVector{1, Float64}), 0., 0.)

mutable struct Particle0d3v <: AbstractParticle
    v::SVector{3, Float64}
    t::Float64
    tau::Float64
end

Base.zero(::Union{Type{Particle0d3v}, Particle0d3v}) = Particle0d3v(zero(SVector{3, Float64}), 0., 0.)

struct ParticleEnsemble{ParticleType} <: AbstractEnsemble
    name::String
    m::Float64
    q::Float64
    coords::Array{ParticleType, 1}
end

Base.getindex(particles::ParticleEnsemble{T}, i::Integer) where T = particles.coords[i]

ParticleEnsemble(name, m, q, n, ParticleType = Particle0d3v) = ParticleEnsemble{ParticleType}(
        name, m, q,
        [zero(ParticleType) for i in 1:n]
    )

    struct NeutralEnsemble <: AbstractEnsemble
    name::String
    m::Float64
    q::Float64
    T::Float64
    n::Float64
    vthermal::Float64
end

NeutralEnsemble(name, T, m, n) = NeutralEnsemble(name, m, 0, T, n, sqrt(k_B*T/m))

@inline function random_sample(s::NeutralEnsemble)
    random_maxwell_v(s.vthermal)
end

function init_monoenergetic(particles::ParticleEnsemble, Eev)
    vtot = sqrt(2*Eev*q_e/particles.m)
    println(vtot)
    for p in particles.coords
        p.v = random_direction(vtot)
    end
end

function init_thermal(particles::ParticleEnsemble, T)
    vthermal = sqrt(k_B*T/particles.m)
    for p in particles.coords
        p.v = random_maxwell_v(vthermal)
    end
end

function init_time(particles::ParticleEnsemble)
    for p in particles.coords
        p.t = 0.
    end
end

function energy(species::AbstractEnsemble, particle::AbstractParticle)
    0.5*species.m*sum(particle.v.^2)
end

function energy(particles::ParticleEnsemble)
    map(p -> energy(particles, p), particles.coords)
end
