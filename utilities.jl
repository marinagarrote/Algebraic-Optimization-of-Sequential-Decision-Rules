
using LinearAlgebra, HomotopyContinuation
using JuMP, Ipopt, Distributions


#Reward function
function discountedReward(Mπ, Mα, Mβ, γ, μ, r)
    Mτ = Mπ * Mβ
    pπ = [transpose(Mτ[:,s_old]) * Mα[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    rπ = diag(r * Mπ * Mβ)
    counter = det(I - γ * pπ + (μ*transpose(rπ)))
    denominator = det(I - γ * pπ)
    return (1-γ) * counter / denominator - (1 - γ)
end

#Check, whether this gives us the right things!
function pmatrix(Mπ, Mα, Mβ)
    #Compute statistical map from states to states given π, β.
    nS = size(Mβ)[2]
    Mτ = Mπ * Mβ
    pπ = [transpose(Mτ[:,s_old])*Mα[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    return pπ
end

function rewardFunction(Mα, Mβ, γ, μ, r)
    (nS, nA) = size(Mα)[2:3]   #Number of states
    @polyvar x[1:nA, 1:nO]  
    Mπ = x

    pπ = pmatrix(Mπ, Mα, Mβ)
    product = r * Mπ * Mβ
    rπ = [product[i, i] for i in 1:nS]

    #ATTENTION: there could be problems depending on whether μ is a vector
    counter = det(I - γ * pπ + vec(μ)*transpose(rπ))
    denominator = det(I - γ * pπ)
    return counter, denominator, Mπ
end

function differentialRewardFunction(Mα, Mβ, γ, μ, r)
    counter, denominator, Mπ = rewardFunction(Mα, Mβ, γ, μ, r)
    return differentiate(counter/denominator, Mπ)
end

function restrict_to_face(counter, denominator, Mπ, F)
    n, m = size(F)
    @var θ[1:n, 1:m]

    new_vars = θ.*F

    for col_index in 1:m
        last_nonzero = (1:n)[F[:, col_index].==1][end]
        new_vars[last_nonzero, col_index] = 1 - sum( new_vars[1:last_nonzero-1, col_index]  )
    end
    vars = reshape(Mπ, n*m)
    nvars = reshape(new_vars, n*m)

    ncounter = subs(counter, reshape(Mπ, n*m) => reshape(new_vars, n*m))
    ndenominator = subs(denominator, reshape(Mπ, n*m) => reshape(new_vars, n*m))
    return ncounter, ndenominator
end


##### Read Grids #####

function observationsGrid(grid, actions)
    (n, m) = size(grid)
    states = findall(grid .> 0)
    nS = length(states)
    nA = length(actions)
    observ = map(i -> map(j -> grid[states[i][1] + actions[j][1], states[i][2] + actions[j][2]] , 1:nA), 1:nS)
    return [states, observ], unique(observ)
end

function observationKernel(grid, actions)
    (SO, observ) = observationsGrid(grid, actions)
    nS = length(SO[1])
    nO = length(observ)
    Mβ = zeros(nO, nS)
    for i in 1:nS
        Mβ[findall(x -> x == SO[2][i] , observ)[1], i] = 1
    end
    return Mβ
end

function transitionKernel(grid, actions, rew, loop=false)
    (SO, observ) = observationsGrid(grid, actions)
    nS = length(SO[1])
    nA = length(actions)
    rew_positions = findall(rew .> 0)

    Mα = reshape([0.0 for i=1:(nS*nS*nA)], nS, nS, nA);

    for s in 1:nS
        for a in 1:nA
            x = SO[1][s][1]
            y = SO[1][s][2]
            if loop
                if CartesianIndex(x,y) in rew_positions
                    Mα[:,s,:] = ones(nS) * transpose(ones(nA)) / nS
                else
                    x += actions[a][1]
                    y += actions[a][2]
                    if grid[x,y] == 1
                        s2 = findall(i -> i == CartesianIndex(x,y), SO[1])[1]
                        Mα[s2,s,a] = 1
                    end
                end
            else
                x += actions[a][1]
                y += actions[a][2]
                if grid[x, y] == 1
                    s2 = findall(i -> i == CartesianIndex(x,y), SO[1])[1]
                   Mα[s2,s,a] = 1
                end
            end
        end
    end
    return Mα
end

function instReward(grid, actions, rew)
    states = findall(grid .> 0)
    return [rew[s] for s in states] * transpose(ones(nA))
end


##### Translation to η #####

### Objective Function
function ObjectiveFunction(η, r)
    (nS, nA) = size(r)
    sum(η[s, a]r[s, a] for a in 1:nA, s in 1:nS)
end

### Linear equalities
function linearEqualitiesPOMDP(η, γ, μ, Mα) 
    nS = size(Mα)[1]
    linEq = [expand(sum(η[s, :]) - γ*(dot(η[:, :], Mα[s, :, :])) - (1-γ)*μ[s]) for s in 1:nS]
    if γ == 1
        linEq = append!(linEq, sum(η) - 1)
    end
    return linEq
end
function linearEqualities(η, γ, μ, Mα) 
    nS = size(Mα)[1]
    linEq = [sum(η[s, :]) - γ*(dot(η[:, :], Mα[s, :, :])) - (1-γ)*μ[s] for s in 1:nS]
    if γ == 1
        linEq = append!(linEq, sum(η) - 1)
    end
    return linEq
end

### Polynomial equalities
function polynomialEqualitiesPOMDP(η, Mβ, nA, deterministic=true)
    ### Check whether β is deterministic
    if sum(Mβ.==1) != size(Mβ)[2]
        println("method only implemented for deterministic observations")
        return
    end

    polEq = []
    Os = findall(sum(eachcol(Mβ)) .> 1)
    for o in Os
        compatibleStates = findall(Mβ[o,:] .> 0)
        sₒ = compatibleStates[1]
        Sₒ = setdiff(compatibleStates, sₒ)
        for s in Sₒ
            for a in 1:(nA-1)
                eq = η[s,a]*sum(η[sₒ,:]) - η[sₒ,a]*sum(η[s, :])
                polEq = vcat(polEq, eq)
            end
        end
    end
    return polEq
end


### Linear inequalities
function linearInequalities(η)
    nS,nA = size(η)
    return reshape(η, nS*nA, 1)
end

### Polynomial inequalities
function polynomialInequalities(η, Mβ, injective=true)
    nO = size(Mβ)[1]
    if rank(Mβ) < nO
        print("observation mechanism does not satisfy the rank condition")
        return
    end
    nA = size(η)[2]
    polIneq = []
    MβInv = inv(Mβ)
    for o in 1:nO
        for a in 1:nA
            p = sum([MβInv[s,o]*η[s,a]*prod([sum(η[s0,:]) for s0 in setdiff(Sₒ, s)]) for s in Sₒ])
            polIneq = vcat(polIneq, p)
        end
    end
    return polIneq
end

### Compute the observation policy from the state action distribution
function observationPolicy(η, Mβ, nS, nA)
    nO = size(Mβ)[1]
    ### Compute the state policy
    τ = zeros((nS, nA));
    for i = 1:nS
        τ[i, :] = η[i, :] / sum(η[i, :]);
    end
    ### Compute the observation policy
    π = transpose(τ) * pinv(Mβ)
    return π
end

function logbarrierSystem(Mα, Mβ, r, γ, μ)
    nO, nS = size(Mβ)
    nA = size(r)[2]

    @var η[1:nS, 1:nA];
    #Define the conditions
    lEqs = linearEqualitiesPOMDP(η, γ, μ, Mα);
    pEqs = unique(polynomialEqualitiesPOMDP(η, Mβ, nA));
    h = vcat(lEqs, pEqs);
    ∇h = differentiate(Vector{Expression}(h), reshape(η, nS*nA));

    @var λ[1:length(h)], t₀;
    #Define the system
    S = System(vcat(reshape(η, nS*nA).*reshape(r, nS*nA) + ones(nS*nA) / t₀ + reshape(η, nS*nA).*(∇h'*λ), h))
    return S
end

function logbarrierSolve_copy(S, r, t, nS, nA)
    S = System(subs.(S.expressions, S.variables[1]=>t))
    #Solve
    sols = HomotopyContinuation.solve(S, show_progress=false);
    #Find the feasible solutions
    rs = solutions(sols, only_nonsingular=false)

    if length(rs) > 0
        rs_nonSing = solutions(sols, only_nonsingular=true)
        rs_sing = setdiff(rs, rs_nonSing)

        nC_total = length(rs)
        nC_nonSing = length(rs_nonSing)
        nC_sing = length(rs_sing)

        if length(rs_nonSing) > 0
            ηs_nonSing = [rs_nonSing[i][1:nS*nA] for i in 1:length(rs_nonSing)]
            ηs_nonSing = [round.(ηs_nonSing[i]; digits=9) for i in 1:length(ηs_nonSing)]
            ηs_real_nonSing = ηs_nonSing[isreal.(ηs_nonSing)]

            nη_nonSing = length(unique(ηs_nonSing))
            nRealη_nonSing = length(unique(ηs_real_nonSing))
        else
            nη_nonSing = 0
            nRealη_nonSing = 0
        end

        if length(rs_sing) > 0
            ηs_sing = [rs_sing[i][1:nS*nA] for i in 1:length(rs_sing)]
            ηs_sing = [round.(ηs_sing[i]; digits=9) for i in 1:length(ηs_sing)]
            ηs_real_sing = ηs_sing[isreal.(ηs_sing)]

            nη_sing = length(unique(ηs_sing))
            nRealη_sing = length(unique(ηs_real_sing))
        else
            nη_sing = 0
            nRealη_sing = 0
        end

        ηs = [rs[i][1:nS*nA] for i in 1:length(rs)]
        ηs = [round.(ηs[i]; digits=9) for i in 1:length(ηs)]
        ηs_real = ηs[isreal.(ηs)]

        nη_total = length(unique(ηs))
        nRealη_total = length(unique(ηs_real))

        # Check the dual feasability
        ηs_real = filter(r -> all(Float64.(r).>-10^-10), ηs_real)
        ηs_positive = length(ηs_real)
        ηKKT = ηs_real[argmax([Float64(transpose(reshape(r, nS*nA))*ηs_real[i]) for i in 1:length(ηs_real)])]

        return (reshape(Float64.(ηKKT), (nS, nA)), nC_total, nη_total, nRealη_total, nC_nonSing, nη_nonSing, nRealη_nonSing, nC_sing, nη_sing, nRealη_sing)
    end
    return []
end

function logbarrierSolve(S, r, t, nS, nA)
    S = System(subs.(S.expressions, S.variables[1]=>t))
    #Solve
    rs = HomotopyContinuation.solve(S, show_progress=false)
    sols = solutions(rs, only_nonsingular=false)

    if length(sols) > 0

        sols_nonSing = solutions(rs, only_nonsingular=true)
        sols_sing = setdiff(sols, sols_nonSing)

        ηs_nonSing = [sols_nonSing[i][1:nS*nA] for i in 1:length(sols_nonSing)]
        ηs_sing = [sols_sing[i][1:nS*nA] for i in 1:length(sols_sing)]

        (n_nonSing, n_nonSing_real, n_nonSing_pos, n_sing, n_sing_real, n_sing_pos,
        η_nonSing_max, r_nonSing_max, η_sing_max, r_sing_max, maxη) = analyzeSolutions_η(ηs_nonSing, ηs_sing, reshape(r, nS*nA))

        return (length(unique(sols)), n_nonSing, n_nonSing_real, n_nonSing_pos, n_sing, n_sing_real, n_sing_pos,
        η_nonSing_max, r_nonSing_max, η_sing_max, r_sing_max, maxη)
    end
    return []
end


function IpoptSolve(Mα, Mβ, r, γ, μ)
    nO, nS = size(Mβ)
    nA = size(r)[2]
    model = Model(optimizer_with_attributes(Ipopt.Optimizer))
    set_silent(model);
    @variable(model, η[1:nS,1:nA]>=0);
    lEqs = linearEqualities(η, γ, μ, Mα)
    pEqs = polynomialEqualitiesPOMDP(η, Mβ, nA)
    h = vcat(lEqs, pEqs);
    for i in 1:length(h)
        p = h[i]
        @constraint(model, p == 0)
    end
    @NLobjective(model, Max, sum(η[s,a]*r[s,a] for a in 1:nA, s in 1:nS))
    optimize!(model)
    #Access the optimizer and compute the observation policy
    ηIpopt = JuMP.value.(η);
end

function KKTSystem(Mα, Mβ, r, γ, μ, reduceIneq=false)
    nO, nS = size(Mβ)
    nA = size(r)[2]

    @var η[1:nS, 1:nA];
    lEqs = linearEqualitiesPOMDP(η, γ, μ, Mα)
    pEqs = polynomialEqualitiesPOMDP(η, Mβ, nA)
    h = Vector{Expression}(vcat(lEqs, pEqs));
    ∇h = differentiate(h, reshape(η, nS*nA));

    f = ObjectiveFunction(η, r)
    ∇f = reshape(r, nS*nA)

    @var λ[1:length(h)]
    @var κ[1:nS, 1:nA]

    if reduceIneq
        Os = findall(sum(eachcol(Mβ)) .> 1)
        for o in Os
            compatibleStates = findall(Mβ[o,:] .> 0)
            sₒ = compatibleStates[1]
            Sₒ = setdiff(compatibleStates, sₒ)
            for s in Sₒ
                κ = subs(κ, variables(κ[s,:]) => zeros(nA))
            end
        end
    end

    S = System(vcat(∇f + reshape(κ, nS*nA) + ∇h'*λ, h, reshape(κ.*η, nS*nA)))
    return(S)
end

function KKTSolve_copy(Mα, Mβ, r, γ, μ, reduceIneq=false)
    nO, nS = size(Mβ)
    nA = size(r)[2]

    @var η[1:nS, 1:nA];
    lEqs = linearEqualitiesPOMDP(η, γ, μ, Mα)
    pEqs = polynomialEqualitiesPOMDP(η, Mβ, nA)
    h = Vector{Expression}(vcat(lEqs, pEqs));
    ∇h = differentiate(h, reshape(η, nS*nA));

    f = ObjectiveFunction(η, r)
    ∇f = reshape(r, nS*nA)

    @var λ[1:length(h)]
    @var κ[1:nS, 1:nA]

    if reduceIneq
        Os = findall(sum(eachcol(Mβ)) .> 1)
        for o in Os
            compatibleStates = findall(Mβ[o,:] .> 0)
            sₒ = compatibleStates[1]
            Sₒ = setdiff(compatibleStates, sₒ)
            for s in Sₒ
                κ = subs(κ, variables(κ[s,:]) => zeros(nA))
            end
        end
    end

    S = System(vcat(∇f + reshape(κ, nS*nA) + ∇h'*λ, h, reshape(κ.*η, nS*nA)))
    #Solve
    sols = HomotopyContinuation.solve(S, show_progress=false)

    rs = solutions(sols, only_nonsingular=false)

    if length(rs) > 0
        rs_nonSing = solutions(sols, only_nonsingular=true)
        rs_sing = setdiff(rs, rs_nonSing)

        nC_total = length(rs)
        nC_nonSing = length(rs_nonSing)
        nC_sing = length(rs_sing)

        if length(rs_nonSing) > 0
            ηs_nonSing = [rs_nonSing[i][1:nS*nA] for i in 1:length(rs_nonSing)]
            ηs_nonSing = [round.(ηs_nonSing[i]; digits=9) for i in 1:length(ηs_nonSing)]
            ηs_real_nonSing = ηs_nonSing[isreal.(ηs_nonSing)]

            nη_nonSing = length(unique(ηs_nonSing))
            nRealη_nonSing = length(unique(ηs_real_nonSing))
        else
            nη_nonSing = 0
            nRealη_nonSing = 0
        end

        if length(rs_sing) > 0
            ηs_sing = [rs_sing[i][1:nS*nA] for i in 1:length(rs_sing)]
            ηs_sing = [round.(ηs_sing[i]; digits=9) for i in 1:length(ηs_sing)]
            ηs_real_sing = ηs_sing[isreal.(ηs_sing)]

            nη_sing = length(unique(ηs_sing))
            nRealη_sing = length(unique(ηs_real_sing))
        else
            nη_sing = 0
            nRealη_sing = 0
        end

        ηs = [rs[i][1:nS*nA] for i in 1:length(rs)]
        ηs = [round.(ηs[i]; digits=9) for i in 1:length(ηs)]
        ηs_real = ηs[isreal.(ηs)]

        nη_total = length(unique(ηs))
        nRealη_total = length(unique(ηs_real))

        # Check the dual feasability
        ηs_real = filter(r -> all(Float64.(r).>-10^-10), ηs_real)
        ηs_positive = length(ηs_real)
        ηKKT = ηs_real[argmax([Float64(transpose(reshape(r, nS*nA))*ηs_real[i]) for i in 1:length(ηs_real)])]

        return (reshape(Float64.(ηKKT), (nS, nA)), nC_total, nη_total, nRealη_total, nC_nonSing, nη_nonSing, nRealη_nonSing, nC_sing, nη_sing, nRealη_sing)
    end
    return []
end

function KKTSolve(Mα, Mβ, r, γ, μ, reduceIneq=false)
    nO, nS = size(Mβ)
    nA = size(r)[2]

    @var η[1:nS, 1:nA];
    lEqs = linearEqualitiesPOMDP(η, γ, μ, Mα)
    pEqs = polynomialEqualitiesPOMDP(η, Mβ, nA)
    h = Vector{Expression}(vcat(lEqs, pEqs));
    ∇h = differentiate(h, reshape(η, nS*nA));

    f = ObjectiveFunction(η, r)
    ∇f = reshape(r, nS*nA)

    @var λ[1:length(h)]
    @var κ[1:nS, 1:nA]

    if reduceIneq
        Os = findall(sum(eachcol(Mβ)) .> 1)
        for o in Os
            compatibleStates = findall(Mβ[o,:] .> 0)
            sₒ = compatibleStates[1]
            Sₒ = setdiff(compatibleStates, sₒ)
            for s in Sₒ
                κ = subs(κ, variables(κ[s,:]) => zeros(nA))
            end
        end
    end

    S = System(vcat(∇f + reshape(κ, nS*nA) + ∇h'*λ, h, reshape(κ.*η, nS*nA)))
    #Solve
    rs = HomotopyContinuation.solve(S, show_progress=false)
    sols = solutions(rs, only_nonsingular=false)

    if length(sols) > 0

        sols_nonSing = solutions(rs, only_nonsingular=true)
        sols_sing = setdiff(sols, sols_nonSing)

        ηs_nonSing = [sols_nonSing[i][1:nS*nA] for i in 1:length(sols_nonSing)]
        ηs_sing = [sols_sing[i][1:nS*nA] for i in 1:length(sols_sing)]

        (n_nonSing, n_nonSing_real, n_nonSing_pos, n_sing, n_sing_real, n_sing_pos,
        η_nonSing_max, r_nonSing_max, η_sing_max, r_sing_max, maxη) = analyzeSolutions_η(ηs_nonSing, ηs_sing, reshape(r, nS*nA))

        return (length(unique(sols)), n_nonSing, n_nonSing_real, n_nonSing_pos, n_sing, n_sing_real, n_sing_pos,
        η_nonSing_max, r_nonSing_max, η_sing_max, r_sing_max, maxη)
    end
    return []
end

function lagrangeSystem(f₀, h₀)
    h₀ = filter(r -> r != 0.0, h₀)

    @var λ₀[1:length(h₀)]
    intVars = variables(h₀)

    ∇f₀ = differentiate(f₀, intVars)
    ∇h₀ = differentiate(h₀, intVars)
    S = System(vcat(∇f₀ + ∇h₀'*λ₀, h₀))
end

function uniqueEqs(h₀)
    h₀ = expand.(h₀)
    uniq_h₀ = []
    for i in 1:length(h₀)
        if !(issubset(h₀[i], uniq_h₀)) & !(issubset(expand(-h₀[i]), uniq_h₀))
            uniq_h₀ = vcat(uniq_h₀, h₀[i])
        end
    end
    Vector{Expression}(uniq_h₀)
end


function analyzeSolutions_η(ηs_nonSing, ηs_sing, r_vec)

    ηs_nonSing = [round.(Vector{ComplexF64}(ηs_nonSing[ii]); digits=9) for ii in 1:length(ηs_nonSing)]
    ηs_nonSing_real = ηs_nonSing[isreal.(ηs_nonSing)]
    ηs_nonSing_pos = filter(r -> all(Float64.(r).>-10^-10), ηs_nonSing_real)

    ηs_sing = [round.(Vector{ComplexF64}(ηs_sing[ii]); digits=9) for ii in 1:length(ηs_sing)]
    ηs_sing_real = ηs_sing[isreal.(ηs_sing)]
    ηs_sing_pos = filter(r -> all(Float64.(r).>-10^-10), ηs_sing_real)

    if length(ηs_nonSing_pos) > 0
        η_nonSing_max = ηs_nonSing_pos[argmax([Float64(transpose(r_vec)*ηs_nonSing_pos[i]) for i in 1:length(ηs_nonSing_pos)])]
        r_nonSing_max = maximum([Float64(transpose(r_vec)*ηs_nonSing_pos[i]) for i in 1:length(ηs_nonSing_pos)])
    else
        η_nonSing_max = Vector{Union{Float64,Missing}}(missing, length(r_vec))
        r_nonSing_max = -Inf
    end

    if length(ηs_sing_pos) > 0
        η_sing_max = ηs_sing_pos[argmax([Float64(transpose(r_vec)*ηs_sing_pos[i]) for i in 1:length(ηs_sing_pos)])]
        r_sing_max = maximum([Float64(transpose(r_vec)*ηs_sing_pos[i]) for i in 1:length(ηs_sing_pos)])
    else
        η_sing_max = Vector{Union{Float64,Missing}}(missing, length(r_vec))
        r_sing_max = -Inf
    end


    if r_nonSing_max >= r_sing_max
        maxη = "nonSingular"
    else
        maxη = "singular"
    end

    return([length(unique(ηs_nonSing)), length(unique(ηs_nonSing_real)), length(unique(ηs_nonSing_pos)),
            length(unique(ηs_sing)), length(unique(ηs_sing_real)), length(unique(ηs_sing_pos)),
            η_nonSing_max, r_nonSing_max, η_sing_max, r_sing_max, maxη])
end
