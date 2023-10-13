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

@testset "particle_tests" begin

        oxygen_test_particle = MonteCarloCollisions.Particles("oxygen_test", 16*amu, q_e, 2)
    
        fluor_test_particle = MonteCarloCollisions.Particles("Fluor", 19*amu, q_e, 2)
    
        sodium_test_particle = MonteCarloCollisions.Particles("sodium_test_particle", 23*amu, q_e, 2)
    
        MonteCarloCollisions.init_thermal(sodium_test_particle, 9)
    
        MonteCarloCollisions.init_thermal(fluor_test_particle, 10)
    
        MonteCarloCollisions.init_thermal(oxygen_test_particle, 11)
    
        scatter_v = MonteCarloCollisions.scatter(sodium_test_particle.list[1].v, fluor_test_particle.list[1].v, sodium_test_particle.m, fluor_test_particle.m, norm(sodium_test_particle.list[1].v - fluor_test_particle.list[1].v), 0)
            
        @test fluor_test_particle.list[1].v != scatter_v

        @test MonteCarloCollisions.energy(sodium_test_particle) > MonteCarloCollisions.energy(fluor_test_particle) > MonteCarloCollisions.energy(oxygen_test_particle)

        @test norm(reference_particle.list[1].v) != norm(scatter_v)

end

@testset "maxwell_tests" begin
<<<<<<< HEAD
    nsampl = 1000000
    
    oxygen_test = oxygen_test = Neutrals("Oxygen", 300., 16*amu, 2e19)
    
    std_sample = [norm(random_maxwell_v(oxygen_test.vthermal)) for x in 1:nsampl]
    
    test_sample = [norm(random_maxwell_v(oxygen_test.vthermal)) for x in 1:nsampl]
    
    mean_test = [random_maxwell_v(oxygen_test.vthermal) for x in 1:nsampl]
    
    @test sqrt(8/pi)*oxygen_test.vthermal - std(std_sample)/100 *5 < mean(test_sample)# < sqrt(8/pi)*oxygen_test.vthermal + std(std_sample)/100
    
    @test mean(test_sample) < sqrt(8/pi)*oxygen_test.vthermal + std(std_sample)/100 *5

    @test mean(mean_test) < [0,0,0]
    
    
end 

    
@testset "interaction_tests" begin

    oxygen_test = MonteCarloCollisions.Neutrals("Oxygen", 300., 16*amu, 2e19);
    
    oxygen_list = MonteCarloCollisions.load_interactions_lxcat("test_input.txt", electrons, oxygen_test)

    test_interaction = MonteCarloCollisions.make_interactions(electrons, oxygen_list)

    @test test_interaction.list[1][1].name == "EFFECTIVE O"

    @test length(test_interaction.spec2_names) == 1 
    
    @test typeof(test_interaction.spec2_list[1]) == Neutrals

    @test length(test_interaction.list[1][1].species1.list) == length(electrons.list)

    @test test_interaction.spec2_list[1] == oxygen_test

    @test length(test_interaction.list) == 1

    svmax_find!(test_interaction, 20.)

    init_rates!(test_interaction)

    @test length(test_interaction.svmax_list) == 1

    @test test_interaction.svmax_list[1] == 4.930053435374193e-13

    @test test_interaction.prob_list[1] == 1.0

end


