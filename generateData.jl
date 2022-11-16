using JuMP
import Ipopt
using Distributions
using Dates
using JLD2
include("utilities.jl")
folder = pwd()

actions_states = [2 3; 3 4; 3 5]
D = [[[3], [2 1], [1 1 1]], [[4], [3 1], [2 2], [2 1 1], [1 1 1 1]], [[5], [3 2], [4 1], [3,1,1], [2,2,1], [2,1,1,1], [1,1,1,1,1]]]
iterations = 20

for i in 1:size(actions_states)[1]
    (nA, nS) =  actions_states[i,:]
    for j in 1:length(D[i])
        dₒ = D[i][j]
        nO = length(dₒ)
        for it in 1:iterations
            # Generate random data
            begin
                Mβ_found = false
                Mβ = zeros(nO, nS);
                while !Mβ_found
                    for s in 1:nS
                        v = randn(nO)
                        Mβ[:,s] = v .== maximum(v)
                    end
                    if size(Mβ[[Mβ[o,:] != zeros(nS) for o in 1:nO],:])[1] == size(Mβ)[1] &&
                        length(setdiff(sum(eachcol(Mβ)), dₒ)) == 0 &&
                        length(sum(eachcol(Mβ))) == length(dₒ)
                        Mβ_found = true
                    end
                end
                Mα = zeros(nS, nS, nA);
                for s in 1:nS
                    for a in 1:nA
                        Mα[:,s,a] = rand(Dirichlet(nS, 1))
                    end
                end
                μ = rand(Dirichlet(nS, 1));
                γ = 0.9
                r = randn(nS, nA);
            end

            # Save data
            save_object(string(folder,"/Data/beta_", nA, "_", nS, "_", nO, "_", j, "_", it), Mβ)
            save_object(string(folder,"/Data/alpha_", nA, "_", nS, "_", nO, "_", j, "_", it), Mα)
            save_object(string(folder,"/Data/r_", nA, "_", nS, "_", nO, "_", j, "_", it), r)
            save_object(string(folder,"/Data/mu_", nA, "_", nS, "_", nO, "_", j, "_", it), μ)
        end
    end
end
