module MonteCarloCollisions

using StaticArrays
using PhysicalConstants.CODATA2018
using Unitful
using Interpolations
using Optim
using Random: rand, randn, randexp
using LinearAlgebra: norm, dot, cross



export NeutralEnsemble, ParticleEnsemble, Interaction, Interactions, add_interaction!
export load_interaction_lxcat, load_interactions_lxcat, svmax_find!, init_rates!, make_interactions
export init_time, init_monoenergetic, init_thermal
export advance, advance!, energy
export m_e, q_e, amu

const m_e = ustrip(u"kg", CODATA2018.m_e)
const amu = ustrip(u"kg", CODATA2018.AtomicMassConstant)
const q_e = ustrip(u"C", CODATA2018.e)
const k_B = ustrip(u"J/K", CODATA2018.k_B)

include("RandomSampling.jl")
include("Particles.jl")

InterpolationType = typeof(interpolate(([0.,1.],), [0.,1.], Gridded(Linear())))

struct Interaction
    name::String
    DE::Float64
    species1::ParticleEnsemble
    species2::NeutralEnsemble
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

"""load cross section from text file from lxcat database
and return Interaction"""
function load_interaction_lxcat(filename, process, species1, species2)
    DE, CS = load_cross_section_lxcat(filename, process)
    Interaction(process, CS[:,2], CS[:,1], DE, species1, species2)
end

"""load cross section from text file from lxcat database
and return DE and tabulated cross section"""
function load_cross_section_lxcat(filename, process)
    nl = "\r\n"
    open(filename) do f
        readuntil(f, "PROCESS: "*process*nl)
        parse_cross_section_lxcat(f, process)
    end
end

function parse_cross_section_lxcat(stream, process)
    elastic_collision_types = ["Elastic", "Effective", "Backscat", "Isotropic"]
    inelastic_collision_types = ["Excitation", "Ionization", "Attachment"]
    nl = "\r\n"

    collision_type = split(process, ", ")[end]
    collision_target = split(process, ", ")[begin]

    header = split(readuntil(stream, "-----"*nl), nl)

    if collision_type in inelastic_collision_types
        DE = NaN64
        for hl in header
            Estr = match(r"E = ([^ ]* )", hl)
            if Estr !== nothing
                DE = parse(Float64, Estr[1])
            end
        end
        if isnan(DE)
            error("DE of interaction "*collision_type*" "*collision_target*" not parsed")
        end
    else
        DE = 0.
    end

    data_string = readuntil(stream, "-----"*nl)
    data_numbers = map(split(data_string, nl)[1:end-1]) do line
        map(x->parse(Float64, x), split(line, "\t"))
    end
    CS = hcat(data_numbers...)'
    DE, CS
end

function load_interactions_lxcat(filename, species1, species2)
    result = Interaction[]
    # load all interactions from text file from lxcat database
    open(filename) do f
        for l in eachline(f)
            if startswith(l, "PROCESS: ")
                process = split(l, "PROCESS: ")[end]
                DE, CS = parse_cross_section_lxcat(f, process)
                push!(result, Interaction(process,
                        CS[:,2], CS[:,1], DE, species1, species2
                    ))
            end
        end
    end
    result
end

mutable struct Interactions{P <: AbstractEnsemble}
    # structure representing all interactions of Species spec1 with
    # partners from list spec2_list
    spec1::P
    list::Vector{Vector{Interaction}} # list[i] contains list of interactions with spec2_list[1]
    spec2_names::Vector{String}
    spec2_list::Vector{NeutralEnsemble} # list of interacting species
    svmax_list::Vector{Float64}  # maximum of total sigma*v_rel for each spec2
    prob_list::Vector{Float64}   # interaction rates of spec2 divided by total rate
    rate::Float64 # (s-1) total interaction rate of spec1
end

Interactions(species::P) where P <: AbstractEnsemble = 
    Interactions(
        species,
        Vector{Interaction}[], String[], NeutralEnsemble[], Float64[], Float64[], 0.)

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

include("Simulation.jl")

end
