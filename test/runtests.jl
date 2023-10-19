using MonteCarloCollisions
using LinearAlgebra: dot
using Statistics
using LinearAlgebra
using StaticArrays
using Test

@testset "MonteCarloCollisions.jl" begin
    nsampl = 1000
    samples = [MonteCarloCollisions.random_direction(1.) for i in 1:nsampl]
    @test max(norm.(samples)...) ≈ min(norm.(samples)...)
    @test mean(samples) < std(samples)/sqrt(nsampl)*5

    # test Boris mover
    electrons = ParticleEnsemble("electron", m_e, q_e, 10);
    nointeraction = make_interactions(electrons, Interactions[])
    init_thermal(electrons, 10000)
    B = SVector(0., 0., 0.01)
    E = SVector(0., 0., 0.)

    steps_per_Tc = 8
    Bstep = 1/steps_per_Tc

    Bqm = B * electrons.q / electrons.m
    omega = norm(Bqm)
    theta = 2*pi*Bstep
    dt = 2*tan(theta/2)/omega


    # test that velocity returns to original after cyclotron period
    v0 = map(x->x.v, electrons.coords)
    advance!(electrons, nointeraction, E, B, dt*steps_per_Tc, Bstep)
    v1 = map(x->x.v, electrons.coords)
    @test all(v0 .≈ v1)

    # velocity oposite after half period
    init_time(electrons)
    v0 = map(x->x.v[1:2], electrons.coords)
    advance!(electrons, nointeraction, E, B, dt*steps_per_Tc/2, Bstep)
    v1 = map(x->x.v[1:2], electrons.coords)
    @test all(v0 .≈ .-v1)

    # velocity perpendicular after quarter period
    init_time(electrons)
    v0 = map(x->x.v[1:2], electrons.coords)
    advance!(electrons, nointeraction, E, B, dt*steps_per_Tc/4, Bstep)
    v1 = map(x->x.v[1:2], electrons.coords)
    @test all(dot(v0 .+ v1, v1) .≈ dot(v1, v1))

    # energy conservation
    init_time(electrons)
    v0 = map(x->x.v[1:2], electrons.coords)
    advance!(electrons, nointeraction, E, B, dt*steps_per_Tc*123.5678, Bstep)
    v1 = map(x->x.v[1:2], electrons.coords)
    @test all(dot(v0, v0) .≈ dot(v1, v1))


    # test data loading
    argon = NeutralEnsemble("argon", 300., 40*amu, 1e22)
    helium = NeutralEnsemble("helium", 300., 4*amu, 1e23)
    electrons = ParticleEnsemble("electron", m_e, q_e, 1000)
    helium_interaction_list = load_interactions_lxcat("../data/CS_e_He.txt", electrons, helium)
    @test length(helium_interaction_list) == 3
    @test length(helium_interaction_list[1].sigmav) == 8
end
