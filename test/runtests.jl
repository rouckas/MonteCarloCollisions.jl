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

    negative_weight_particle = MonteCarloCollisions.Particles("negative_weight", -1, 100, 1)

    reference_particle = MonteCarloCollisions.Particles("reference", 4, 100, 1)

    huge_weight_particle = MonteCarloCollisions.Particles("huge_weight", 5000, 100, 1)

    MonteCarloCollisions.init_monoenergetic(reference_particle, 10)

    MonteCarloCollisions.init_monoenergetic(huge_weight_particle, 10)

    scatter_v = MonteCarloCollisions.scatter(reference_particle.list[1].v, huge_weight_particle.list[1].v, reference_particle.m, huge_weight_particle.m, norm(reference_particle.list[1].v - huge_weight_particle.list[1].v), 0)

    electrons.list[1].t = 8

    init_time(negative_weight_particle)

    @test negative_weight_particle.list[1].t = 0

    @test energy(negative_weight_particle) < energy(reference_particle) < energy(huge_weight_particle)

    @test norm(reference_particle.list[1].v) != norm(scatter_v)

end

@testset "maxwell_tests" begin 

    electrons = MonteCarloCollisions.Particles("electron", m_e, q_e, 10000);

    MonteCarloCollisions.init_thermal(electrons, 100.)

    maxwell_test_list_electrons = []

    for particle in electrons.list
        push!(maxwell_test_list, norm(particle.v))
    end
    
    oxygen_test_list = []



    while length(oxygen_test_list) != 10000
        g = random_maxwell_v(oxygen_test.vthermal)
        push!(oxygen_test_list, norm(g))
    end
    
    @test mean(pxygen_test_list) ≈ 6328.


    @test mean(maxwell_test_list) ≈ 16200.


end
    
@testset "interaction_tests" begin

    oxygen_test = MonteCarloCollisions.Neutrals("Oxygen", 300., 16*amu, 2e19);
    
    oxygen_list = MonteCarloCollisions.load_interactions_lxcat("test2.txt", electrons, oxygen_test)

    test_interaction = MonteCarloCollisions.make_interactions(electrons, oxygen_list)

    @test test_intearction.list[1][1].name == "EFFECTIVE O"

    @test lenght(test_interaction.spec2_names) == 1 
    
    @test typeof(test_interaction.spec2_list[1]) == Neutrals

    @test length(test_interaction.list[1][1].species1.list) == length(electrons.list)

    @test test_interaction.spec2_list[1] == oxygen_test

    @test length(test_interaction.list) == 1

    MonteCarloCollisions.svmax_find!(test_interaction, 20.)

    MonteCarloCollisions.init_rates!(test_intearction)

    @test length(test_intearction.svmax_list) == 1

    @test test_interaction.svmax_list[1] == 4.930053435374193e-13

    @test test_interaction.prob_list[1] == 1.0

end
