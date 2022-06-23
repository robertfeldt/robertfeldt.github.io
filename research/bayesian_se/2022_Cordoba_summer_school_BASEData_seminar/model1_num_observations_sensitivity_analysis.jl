# Goal here is to analyse how sensitive our results are to the 
# number of observations in the underlying data. This is important
# to know since, in this case, when we are evaluating algorithms
# and we can decide how many times to run them we would like to
# run as few times as possible. We can also check if we become
# more sensitive to the prior if we have fewer observations.
# Main idea: Run multiple BDA sampling runs for different size of
# datasets (from few up to the full set) and calculate how the end
# results differ.
include("common.jl")

using ParetoSmooth

# We use the simple m1 again
@model function m1_prior_parameterized(alg1, alg2, twowins, priorExpParam) # priorExpParam is the parameter to the Exponential prior
    # Hyperparam
    s ~ Exponential(priorExpParam)

    # Logits per algorithm
    a ~ filldist(Normal(0, s), N_alg)

    for i in 1:length(twowins)
        # Since we model if alg 2 will "win" that should be the positive logit
        v = logistic(a[alg2[i]] - a[alg1[i]])
        twowins[i] ~ Bernoulli(v)
    end
end;

#
# Prepare data to be input in the model. We prepare the full data
# so that we can later take different subsets of it.
#
df = dffull
Nobs = nrow(df)
const alg1 = Int[findfirst(==(a), Algs) for a in df.Alg1]
const alg2 = Int[findfirst(==(a), Algs) for a in df.Alg2]
const twowins = df.Winner

# Function that samples the posterior based on a subset of all
# observations and a given prior paramater.
function m1_posterior_for_prior(N::Int, prioparam)
    m = m1_prior_parameterized(alg1[1:N], alg2[1:N], twowins[1:N], prioparam)
    chains = sample(m, DynamicNUTS(), MCMCThreads(), 2_000, 5)
    return chains, m
end

# Access Rhat values of chain and basic checks they are ok
rhats(chain) = summarystats(chain)[:, :rhat]
rhat_diffs(chain) = Float64[abs(rh - 1.0) for rh in rhats(chain)]
max_rhat_diff(chain) = maximum(rhat_diffs(chain))
num_bad_rhats(chain) = count(c -> c > 0.01, rhat_diffs(chain))

# Now we loop from 10 observations and up to the full set. Between
# 10 and 100 we take steps of 10 and then steps of 50.
# For each value we also check 4 different prior parameters.
# For each posterior we then calculate some outcome parameters and
# save all results in a new data frame.
# For each chain we also count the number of parameters that had
# bad Rhat values (i.e. out side of (0.99, 1.01)).
num_obs_sequence = vcat(collect(10:10:90), collect(100:50:nrow(df)))
prior_params = [0.1, 0.5, 1.0, 5.0]
res = DataFrame(
    N = Int[],              # Number of data observations
    PriorParam = Float64[], # Prior parameter

    # Info about the sampling and chains and "performance" of the model:
    Time = Float64[],        # Time for the sampling
    NumBadRhat = Int[],      # Number of parameters with "bad" Rhat values, should ideally be 0
    MaxRhatDiff = Float64[], # Largest Rhat diff from 1.0 (large value indicates diverging chain)
    PsisLoo_elpd = Float64[],
    PsisLoo_peff = Float64[],

    # Outcomes:
    prob_ADE_vs_GSS = Float64[], # Probability that adaptive_de wins over generating_set_search
    std_ADE_vs_GSS = Float64[],  # Variation in probability that adaptive_de wins over generating_set_search
)

# Dummy run so things are compiled:
c, m = m1_posterior_for_prior(10, 1.0)
a2 = findfirst(==("adaptive_de_rand_1_bin_radiuslimited"), Algs)
a1 = findfirst(==("generating_set_search"), Algs)

for N in num_obs_sequence
    for p in prior_params
        @info "N = $N, prior param = $p"
        time_start = time()
        c, m = m1_posterior_for_prior(N, p)
        elapsed = time() - time_start

        pl = psis_loo(m ,c)

        # Predict probability that alg2 better than alg1
        preds = predict(m1_prior_parameterized([a1], [a2], [missing], p), c; include_all = true)
        Nparams = 1+N_alg
        ss = summarystats(preds)
        prob_alg2_wins = round(ss[Nparams+1, :mean], digits=2)
        @info "  prob alg2 wins = $prob_alg2_wins"
        std_alg2_wins = round(ss[Nparams+1, :std], digits=2)
        
        push!(res, (N, p, elapsed, 
            num_bad_rhats(c), max_rhat_diff(c),
            pl.estimates(:cv_elpd, :total),
            pl.estimates(:p_eff, :total),
            prob_alg2_wins,
            std_alg2_wins
        ))
    end
end

res |> CSV.write(joinpath(@__DIR__(), "result_num_obs_sensitivity_analysis.csv"))

# Run the following to generate graphs, if you have R, ggplot2, and RCall.jl installed.
include("plot_num_obs_sensitivity_analysis_graphs.jl")