# MonteCarloCollisions

[![Build Status](https://github.com/rouckas/MonteCarloCollisions.jl/workflows/CI/badge.svg)](https://github.com/rouckas/MonteCarloCollisions.jl/actions)
[![Coverage](https://codecov.io/gh/rouckas/MonteCarloCollisions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/rouckas/MonteCarloCollisions.jl)

### Features

- Monte Carlo simulation of particle collisions in a gas or plasma.
- Mainly intended for calculation of electron energy distribution function in low temperature plasma
- Can be easily generalized if needed
- Compatibility tested with Julia 1.6, but Julia version >= 1.7 is recommended for best performance

### Basic usage
- First download the relevant tabulated cross sections from [LXCAT](https://nl.lxcat.net/data/set_type.php) database
- Let's simulate electrons in helium/argon mixture using the cross sections from the Phelps' database at LXCAT - the collision cross sections will be stored in `data/CS_e_He_Phelps.txt` and `data/CS_e_Ar_Phelps.txt`
- Then in Julia
```julia
# import the module
using MonteCarloCollisions
using Statistics, StaticArrays, Plots
#
# set up the background interacting species.
# NeutralEnsemble(name, temperature (K), mass (kg), number density (m^-3)
helium = NeutralEnsemble("helium", 300., 4*amu, 1e23);
argon = NeutralEnsemble("argon", 300., 40*amu, 1e22);
#
# and the primary simulated species
# ParticleEnsemble(name, mass (kg), charge (C), number of particles)
electrons = ParticleEnsemble("electron", m_e, q_e, 10000);
#
# Create `Interactions` object from the cross section data
helium_interaction_list = load_interactions_lxcat("data/CS_e_He_Phelps.txt", electrons, helium);
argon_interaction_list = load_interactions_lxcat("data/CS_e_Ar_Phelps.txt", electrons, argon);
electron_interactions = make_interactions(electrons, vcat(helium_interaction_list, argon_interaction_list));
#
# Start simulation with thermal electrons at T = 10000 K
init_thermal(electrons, 10000)
# reset their simulation time
init_time(electrons)
# set external electrostatic field at 1000 V/m in x direction
E = SVector(1000., 0., 0.)
#
# Currently MonteCarloCollisions provides advance! method to advance the particles in time by a fixed time
# To simulate the time evolution and the equilibrium distribution, we need to write a loop
tax = range(0, 1e-6, length=101)
Emeans = zeros(length(tax))
energies = []
t_equilib = 2e-7
for i in 1:size(tax,1)
    advance!(electrons, electron_interactions, E, tax[i])
    Emeans[i] = mean(energy(electrons))
    if tax[i] >= t_equilib
        append!(energies, energy(electrons))
    end
end
#
# and we can check the convergence rate by looking at the time evolution of the mean energy
plot(tax ./ 1e-6, Emeans ./ q_e, label="", xlabel="t (μs)", ylabel="⟨E⟩ (eV)")
#
# or we can plot the actual equilibrium distribution of electrons
stephist(energies ./ q_e, normalize=:pdf,
        label="",
        yaxis=:log,
        xlabel="E (eV)",
        ylabel="f(E)",
        yticks=10. .^ (-5:-1))
```
