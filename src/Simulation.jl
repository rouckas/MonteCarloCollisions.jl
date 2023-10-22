function advance!(particles::ParticleEnsemble, interactions::Interactions, E, B, tmax::Real, Bstep::Real = 0.1)
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
    for particle in particles.coords
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


function advance!(particles::ParticleEnsemble, interactions::Interactions, E, tmax::Real)
    q = particles.q
    m = particles.m
    Eqm = SVector{3,Float64}(E.*q/m)

    tau_mean = 1/interactions.rate

    Threads.@threads for particle in particles.coords
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
