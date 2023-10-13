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
    electrons = Particles("electron", m_e, q_e, 10);
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
    v0 = map(x->x.v, electrons.list)
    advance!(electrons, nointeraction, E, B, dt*steps_per_Tc, Bstep)
    v1 = map(x->x.v, electrons.list)
    @test all(v0 .≈ v1)

    # velocity oposite after half period
    init_time(electrons)
    v0 = map(x->x.v[1:2], electrons.list)
    advance!(electrons, nointeraction, E, B, dt*steps_per_Tc/2, Bstep)
    v1 = map(x->x.v[1:2], electrons.list)
    @test all(v0 .≈ .-v1)

    # velocity perpendicular after quarter period
    init_time(electrons)
    v0 = map(x->x.v[1:2], electrons.list)
    advance!(electrons, nointeraction, E, B, dt*steps_per_Tc/4, Bstep)
    v1 = map(x->x.v[1:2], electrons.list)
    @test all(dot(v0 .+ v1, v1) .≈ dot(v1, v1))

    # energy conservation
    init_time(electrons)
    v0 = map(x->x.v[1:2], electrons.list)
    advance!(electrons, nointeraction, E, B, dt*steps_per_Tc*123.5678, Bstep)
    v1 = map(x->x.v[1:2], electrons.list)
    @test all(dot(v0, v0) .≈ dot(v1, v1))
end
