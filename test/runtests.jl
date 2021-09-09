using MonteCarloCollisions
using Statistics
using LinearAlgebra
using Test

@testset "MonteCarloCollisions.jl" begin
    nsampl = 1000
    samples = [MonteCarloCollisions.random_direction(1.) for i in 1:nsampl]
    @test max(norm.(samples)...) ≈ min(norm.(samples)...)
    @test mean(samples) < std(samples)/sqrt(nsampl)*5
    # Write your tests here.
end

@testset "particle_tests" begin

        oxygen_test_particle = MonteCarloCollisions.Particles("oxygen_test", 16*amu, q_e, 2)
    
        fluor_test_particle = MonteCarloCollisions.Particles("Fluor", 19*amu, q_e, 2)
    
        sodium_test_particle = MonteCarloCollisions.Particles("sodium_test_particle", 23*amu, q_e, 2)
    
        MonteCarloCollisions.init_thermal(sodium_test_particle, 10)
    
        MonteCarloCollisions.init_thermal(fluor_test_particle, 10)
    
        MonteCarloCollisions.init_thermal(oxygen_test_particle, 10)
    
        scatter_v = MonteCarloCollisions.scatter(sodium_test_particle.list[1].v, fluor_test_particle.list[1].v, sodium_test_particle.m, fluor_test_particle.m, norm(sodium_test_particle.list[1].v - fluor_test_particle.list[1].v), 0)
            
        @test fluor_test_particle.list[1].v != scatter_v

        @test energy(sodium_test_particle) > energy(fluor_test_particle) > energy(oxygen_test_particle)

        @test norm(reference_particle.list[1].v) != norm(scatter_v)

end

    
@testset "interaction_tests" begin

    oxygen_test = MonteCarloCollisions.Neutrals("Oxygen", 300., 16*amu, 2e19);
    
    oxygen_list = MonteCarloCollisions.load_interactions_lxcat("test2.txt", electrons, oxygen_test)

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


