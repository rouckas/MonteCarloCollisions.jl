module MonteCarloCollisions

using StaticArrays
using PhysicalConstants.CODATA2018
using Unitful
using Interpolations
using Optim
using Random: rand, randn, randexp
using LinearAlgebra: norm, dot, cross



export Neutrals, Particles, Interaction, Interactions, add_interaction!
export load_interaction_lxcat, load_interactions_lxcat, svmax_find!, init_rates!, make_interactions
export init_time, init_monoenergetic, init_thermal
export advance, advance!, energy
export m_e, q_e, amu

const m_e = ustrip(u"kg", CODATA2018.m_e)
const amu = ustrip(u"kg", CODATA2018.AtomicMassConstant)
const q_e = ustrip(u"C", CODATA2018.e)
const k_B = ustrip(u"J/K", CODATA2018.k_B)

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

abstract type Species end

struct Neutrals <: Species
    name::String
    T::Float64
    m::Float64
    n::Float64
    vthermal::Float64
end

Neutrals(name, T, m, n) = Neutrals(name, T, m, n, sqrt(k_B*T/m))

@inline function random_sample(s::Neutrals)
    random_maxwell_v(s.vthermal)
end

mutable struct Particle
    v::SVector{3, Float64}
    t::Float64
    tau::Float64
end

struct Particles <: Species
    name::String
    m::Float64
    q::Float64
    n::Int64
    list::Array{Particle, 1}
end

function Particles(name, m, q, n)
    Particles(
        name, m, q, n,
        [Particle(SVector(0.,0.,0.), 0., 0.) for i in 1:n]
    )
end

function init_monoenergetic(particles::Particles, Eev)
    vtot = sqrt(2*Eev*particles.q/particles.m)
    for p in particles.list
        p.v = random_direction(vtot)
    end
end

function init_thermal(particles::Particles, T)
    vthermal = sqrt(k_B*T/particles.m)
    for p in particles.list
        p.v = random_maxwell_v(vthermal)
    end
end

function init_time(particles::Particles)
    for p in particles.list
        p.t = 0.
    end
end

function energy(species::Species, particle::Particle)
    0.5*species.m*sum(particle.v.^2)
end

function energy(particles::Particles)
    map(p -> energy(particles, p), particles.list)
end

InterpolationType = typeof(interpolate(([0.,1.],), [0.,1.], Gridded(Linear())))

struct Interaction
    name::String
    DE::Float64
    species1::Particles
    species2::Neutrals
    mu::Float64
    sigmav::InterpolationType
    #Interpolations.GriddedInterpolation{Float64, 1, Float64, Gridded{Linear{Interpolations.Throw{Interpolations.OnGrid}}}, Tuple{Vector{Float64}}}
    vthresh::Float64
end


import Interpolations: interpolate, Gridded, Linear
function Interaction(name, sigma, E, DE, species1, species2)
    mu = species1.m*species2.m / (species1.m + species2.m)

    v = sqrt.((2*q_e/mu).*E)

    sigmav_interp = interpolate((v,), sigma.*v, Gridded(Linear()))

    vthresh = sigma[2]==0. ? v[2] : v[1]

    Interaction(
        name, DE, species1, species2, mu,
        sigmav_interp, vthresh
    )
end

@inline function sigmav(interaction, v)
    # simple optimization for fast eval of below-treshold collisions
    if v < interaction.vthresh
        sv = 0.
    else
        sv = interaction.sigmav(v)
    end
    sv
end

function load_interaction_lxcat(filename, process, species)
    # load cross section from text file from lxcat database
    nl = "\r\n"
    open(filename) do f
        readuntil(f, process*nl*species*nl)
        DE = parse(Float64, readline(f))

        readuntil(f, "-----"*nl)
        data_string = readuntil(f, "-----"*nl)
        data_numbers = map(split(data_string, nl)[1:end-1]) do line
            map(x->parse(Float64, x), split(line, "\t"))
        end
        DE, hcat(data_numbers...)'
    end
end

function load_interactions_lxcat(filename, species1, species2)
    collision_types = ["ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT"]
    result = Interaction[]
    # load cross section from text file from lxcat database
    nl = "\r\n"
    open(filename) do f
        for l in eachline(f)
            if l in collision_types
                collision_type = l
                collision_target = readline(f)
                if !(collision_type in ["ELASTIC", "EFFECTIVE"])
                    DE = parse(Float64, readline(f))
                else
                    DE = 0.
                end

                readuntil(f, "-----"*nl)
                data_string = readuntil(f, "-----"*nl)
                data_numbers = map(split(data_string, nl)[1:end-1]) do line
                    map(x->parse(Float64, x), split(line, "\t"))
                end
                CS = hcat(data_numbers...)'
                push!(result, Interaction(collision_type*" "*collision_target,
                        CS[:,2], CS[:,1], DE, species1, species2
                    ))
            end
        end
    end
    result
end

mutable struct Interactions
    # structure representing all interactions of Species spec1 with
    # partners from list spec2_list
    spec1::Particles
    list::Vector{Vector{Interaction}} # list[i] contains list of interactions with spec2_list[1]
    spec2_names::Vector{String}
    spec2_list::Vector{Neutrals} # list of interacting species
    svmax_list::Vector{Float64}  # maximum of total sigma*v_rel for each spec2
    prob_list::Vector{Float64}   # interaction rates of spec2 divided by total rate
    rate::Float64 # (s-1) total interaction rate of spec1
end

Interactions(species) = Interactions(species, [], [], [], [], [], 0.)

function add_interaction!(inters, interaction)
    i = findfirst(isequal(interaction.species2.name), inters.spec2_names)
    if isnothing(i)
        push!(inters.list, [interaction])
        push!(inters.spec2_names, interaction.species2.name)
        push!(inters.spec2_list, interaction.species2)
    else
        push!(inters.list[i], interaction)
    end
end

function make_interactions(species, interaction_list, Emax=20.)
    interactions = Interactions(species)
    for interaction in interaction_list
        add_interaction!(interactions, interaction)
    end
    svmax_find!(interactions, Emax)
    init_rates!(interactions)
    interactions
end

function svmax_find!(inters, Emax)
    inters.svmax_list = zeros(length(inters.list))
    inters.prob_list = zeros(length(inters.list))
    for (i, inter) in enumerate(inters.list)
        vmax = sqrt(2*Emax*q_e/inter[1].mu)
        sumv_neg(v) = -sum([sigmav(I, v) for I in inter])
        res = Optim.optimize(sumv_neg, 0, vmax)
        inters.svmax_list[i] = -res.minimum
    end
end

function init_rates!(inters)
    inters.rate = 0
    for (i, svmax) in enumerate(inters.svmax_list)
        inters.prob_list[i] = inters.spec2_list[i].n * svmax
        inters.rate += inters.prob_list[i]
    end
    inters.prob_list ./= inters.rate
end

@inline function scatter(v1, v2, m1, m2, v_rel_norm, DE)
    # transform v1 into center of mass system
    v_cm = (v1*m1 + v2*m2)/(m1 + m2)

    # account for energy loss
    if DE != 0
        mu = m1*m2/(m1+m2)
        E = 0.5*mu*v_rel_norm^2 - DE*q_e
        #if self.collision_type == "ionization": # only the primary electrons are tracked in case of ionization
        #    pE = E*np.random.uniform(0, 1, len(v1))
        #end
        v_rel_norm = sqrt(2*E/mu)
    end

    # random rotation
    v_rel = random_direction(v_rel_norm)
    # reverse transformation
    v1_cm = m2/(m1 + m2) * v_rel
    v1 = v1_cm .+ v_cm
    v1
end

function scatter(v::SVector{3, Float64}, m, inters::Interactions)

    # select colliding species
    i_species = choose(inters.prob_list)
    spec2 = inters.spec2_list[i_species]
    svmax = inters.svmax_list[i_species]
    interlist = inters.list[i_species]

    v_partner = random_sample(spec2)
    v_rel = norm(v-v_partner)

    # select interaction type
    U_interactions = rand()*svmax
    for inter in  interlist
        U_interactions -= sigmav(inter, v_rel)
        if U_interactions < 0
            v = scatter(v, v_partner, m, spec2.m, v_rel, inter.DE)
            break
        end
    end
    v
end

function advance!(particles::Particles, interactions::Interactions, E, B, tmax::Real, Bstep::Real = 0.1)
    q = particles.q
    m = particles.m
    Eqm = SVector{3,Float64}(E.*q/m)
    Bqm = SVector{3,Float64}(B.*q/m)

    # with magnetic field, the maximum time step is a fraction
    # of the cyclotron period T = 2pi/omega.
    omega = norm(Bqm)

    # The desired cyclotron rotation theta in one time step
    # is determined by the parameter Bstep as
    theta = 2*pi*Bstep

    # In the Boris' algorithm, we have for the rotation angle θ
    # tan(θ/2) = -qB/m * Δt/2 = -ω Δt/2
    # Δt = 2 tan(θ/2) /ω
    dtmax = 2*tan(theta/2)/omega

    tau_mean = 1/interactions.rate

    #Threads.@threads
    for particle in particles.list
        t = particle.t
        v = particle.v
        tau = particle.tau

        while true
            tau = randexp()*tau_mean

            t_over = t+tau > tmax
            if t_over
                step = max(tmax - t, 0.)
            else
                step = tau
            end

            dt = step
            while true
                dt = min(step, dtmax)
                v, t = advance(v, t, Eqm, Bqm, dt)
                step -= dt
                if step <= 0
                    break
                end
            end

            if t_over
                break
            end

            v = MonteCarloCollisions.scatter(v, m , interactions,)
        end

        particle.v = v
        particle.t = t
    end
end

function advance!(particles::Particles, interactions::Interactions, E, tmax::Real)
    q = particles.q
    m = particles.m
    Eqm = SVector{3,Float64}(E.*q/m)

    tau_mean = 1/interactions.rate

    Threads.@threads for particle in particles.list
        t = particle.t
        v = particle.v
        tau = particle.tau

        while true
            tau = randexp()*tau_mean

            if t+tau > tmax
                tau = max(tmax - t, 0.)
            end

            v, t = advance(v, t, Eqm, tau)

            if t+tau >= tmax
                break
            end

            v = MonteCarloCollisions.scatter(v, m , interactions,)
        end

        particle.v = v
        particle.t = t
    end
end

@inline function advance(v, t, a, tau)
    v+a*tau, t+tau
end

@inline function advance(v, t, Eqm, Bqm, tau)
    v_minus = v + Eqm * tau/2
    
    T = Bqm * tau/2
    S = T * 2/ (1. + dot(T, T))
    
    v_prime = v_minus + cross(v_minus, T)
    
    v_minus + cross(v_prime, S) + Eqm * tau/2, t+tau
end

end
