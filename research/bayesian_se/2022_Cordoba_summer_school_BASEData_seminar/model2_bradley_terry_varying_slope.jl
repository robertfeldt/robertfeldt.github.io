#include("common.jl")

df = dffull

# Bradley Terry model with 
#   - a single logit value per algorithm, and 
#   - a linear factor capturing scaling with dimension per algorithm
@model function m2(alg1, alg2, logDim, twowins)
    # Hyperparams
    sigma_alg ~ Exponential(1.0)

    # Logits per algorithm
    alg ~ filldist(Normal(0, sigma_alg), N_alg)

    # log(Dim) coefficients per algorithm
    b_alg ~ filldist(Normal(0, 2), N_alg)

    for i in 1:length(twowins)
        a2, a1, logD = alg2[i], alg1[i], logDim[i]
        # Since we model if alg 2 will "win" that should be the positive logit
        v = logistic(alg[a2] + b_alg[a2]*logD - alg[a1] - b_alg[a1]*logD)
        twowins[i] ~ Bernoulli(v)
    end
end;

println("Using $Nobs observations")
alg1 = Int[findfirst(==(a), Algs) for a in df.Alg1]
alg2 = Int[findfirst(==(a), Algs) for a in df.Alg2]
logDims = log10.(df.Dim)
twowins = df.Winner

# TBD: We should do a prior predictive check also here before sampling the posterior...

# Get 10_000 samples from posterior by sampling 2000 from 5 threads:
chains_m2 = sample(m2(alg1, alg2, logDims, twowins), 
    DynamicNUTS(), MCMCThreads(), 2_000, 5
)

# Plot chains and check convergence
plot(chains_m2)

# Get the posterior samples
posterior_m2 = DataFrame(chains_m2)

function summarize_ranks_from_posterior_sample(posterior, logDims)
    as = map(i -> "alg" * "[$i]", 1:length(Algs))
    as_idxs = map(n -> findfirst(==(n), names(posterior)), as)
    bs = map(i -> "b_alg" * "[$i]", 1:length(Algs))
    bs_idxs = map(n -> findfirst(==(n), names(posterior)), bs)

    logits = zeros(Float64, length(Algs))
    ranks = zeros(Int, nrow(posterior)*length(logDims), length(Algs))
    ri = 1
    for i in 1:nrow(posterior)
        for logD in logDims
            for ai in eachindex(Algs)
                a, b = posterior[i, as_idxs[ai]], posterior[i, bs_idxs[ai]]
                logits[ai] = a + b * logD
            end
            ranks[ri, :] = sortperm(collect(logits), rev=true)
            ri += 1
        end
    end

    summarize_ranks(ranks, Algs)
end

println("Ranks based on $Nobs samples:")
ranks_m2 = summarize_ranks_from_posterior_sample(posterior_m2, logDims)

# Simulate other situations such as low-dim and high-dim...

# Only low-dim:
ranks_m2_lowdim = summarize_ranks_from_posterior_sample(posterior_m2, log10.(collect(2:1:20)))
println(ranks_m2_lowdim)

# Only high-dim:
ranks_m2_highdim = summarize_ranks_from_posterior_sample(posterior_m2, log10.(collect(150:10:200)))
println(ranks_m2_highdim)

ranks_m2_lowdim
ranks_m2_highdim

# Let's predict the probabilities of winning for all pairs, in 2 different regimes:
#  low-dim = 10D
#  high-dim = 200D
PredDims = [10, 200]
dfpred = DataFrame(Alg2 = String[], Alg1 = String[], A2 = Int[], A1 = Int[], Dim = Int[])
for a1 in 1:N_alg
    for a2 in (a1+1):N_alg
        for d in PredDims
            push!(dfpred, (Algs[a2], Algs[a1], a2, a1, d))
            push!(dfpred, (Algs[a1], Algs[a2], a1, a2, d))
        end
    end
end
Npreds = nrow(dfpred)
Y = Union{Number, Missing}[missing for _ in 1:Npreds]

preds = predict(m2(dfpred.A1, dfpred.A2, log10.(dfpred.Dim), Y), chains_m2; include_all = true)
Nparams = 1+N_alg+N_alg # sigma and the 5 alphas and the 5 betas
ss = summarystats(preds)
dfpred[!, :Prob] = round.(ss[(Nparams+1):end, :mean], digits=2)
dfpred[!, :Std] = round.(ss[(Nparams+1):end, :std], digits=2)
sort!(dfpred, [:Prob], rev=true)
select(dfpred, [:Alg2, :Alg1, :Dim, :Prob, :Std])

# random_search so bad so let's skip it and show rest only if >=50%
dfpred[dfpred.Prob .>= 0.50 .&& dfpred.Alg1 .!== "random_search", :]

# PSIS-LOO check
using ParetoSmooth
m2_psis_loo = psis_loo(m2(alg1, alg2, logDims, twowins), chains_m2)
#Results of PSIS-LOO-CV with 10000 Monte Carlo samples and 1300 data points. Total Monte Carlo SE of 0.024.
#┌───────────┬─────────┬──────────┬───────┬─────────┐
#│           │   total │ se_total │  mean │ se_mean │
#├───────────┼─────────┼──────────┼───────┼─────────┤
#│   cv_elpd │ -295.05 │    16.28 │ -0.23 │    0.01 │
#│ naive_lpd │ -289.34 │    15.84 │ -0.22 │    0.01 │
#│     p_eff │    5.71 │     0.51 │  0.00 │    0.00 │
#└───────────┴─────────┴──────────┴───────┴─────────┘

# PSIS-LOO compare:
models = (
        m1=m1_psis_loo,
        m2=m2_psis_loo,
);
comps = loo_compare(models)