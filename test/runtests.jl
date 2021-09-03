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

@testset "negative integers - test" begin
    nsampl = 1000
    random_num_array = [ran(r[-1:-10000]) for n in 1:nsampl]
    for x in random_num_array
        variable = random_maxwell_v(x)
        @test variable .< 0
    end
end

@testset "particle_test" begin

    negative_weight_particle = MonteCarloCollisions.Particles("negative_weight", -1, 100, 1)

    reference_particle = MonteCarloCollisions.Particles("reference", 4, 100, 1)

    huge_weight_particle = MonteCarloCollisions.Particles("huge_weight", 5000, 100, 1)

    MonteCarloCollisions.init_monoenergetic(negative_weight_particle.list[1])

    MonteCarloCollisions.init_monoenergetic(reference_particle.list[1])

    MonteCarloCollisions.init_monoenergetic(huge_weight_particle.list[1])

    init_thermal(negative_weight_particle)
    @test negative_weight_particle.list[1].t = 0

    @test energy(negative_weight_particle) < energy(reference_particle) < energy(huge_weight_particle)

end


    

    
