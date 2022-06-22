include("common.jl")

# Bradley Terry model with a single logit value per algorithm
@model function m1(alg1, alg2, twowins)
    # Hyperparam
    s ~ Exponential(1.0)

    # Logits per algorithm
    a ~ filldist(Normal(0, s), N_alg)

    for i in 1:length(twowins)
        # Since we model if alg 2 will "win" that should be the positive logit
        v = logistic(a[alg2[i]] - a[alg1[i]])
        twowins[i] ~ Bernoulli(v)
    end
end;

#
# Prepare data to be input in the model
#
#df = dffull # If want to run with all data
Nobs = nrow(df)
println("Using $Nobs observations")
alg1 = Int[findfirst(==(a), Algs) for a in df.Alg1]
alg2 = Int[findfirst(==(a), Algs) for a in df.Alg2]
twowins = df.Winner

#
# Prior predictive check. We sample using only the prior and then check if looks Plausible?
#
pripd_chains = sample(m1(alg1, alg2, twowins), Prior(), MCMCThreads(), 2_000, 5)

# Check plausibility given some extreme values from the table above.
# We are essentially asking the question: If we compare two algorithms
# at their extreme values is the probability of one "winning" reasonably high?
sum(rand(Bernoulli(logistic(2.88 - (-3.1))), 1000)) # 99.0-99.6% or so. Sounds ok, i.e. a wide prior but not too unlikely, one method can easily be very dominant.

#
# Now condition the model on the data and sample from posterior.
#

# Get 10_000 samples from posterior by sampling 2000 from 5 threads:
chains = sample(m1(alg1, alg2, twowins), DynamicNUTS(), MCMCThreads(), 2_000, 5)

# Plot chains and check convergence. Did the traces mix well and distributions look similar?
plot(chains) # Looks good

# Get the posterior samples so we can compute/simulate with them
posterior_m1 = DataFrame(chains)

# Use the posterior to get ranks which we can then summarize
println("Ranks based on $Nobs samples:")
ranks_m1 = calc_ranks(Algs, posterior_m1, "a")
summarize_ranks(ranks_m1, Algs)

# Let's predict the probabilities of winning for all pairs:
dfpred = DataFrame(Alg2 = String[], Alg1 = String[], A2 = Int[], A1 = Int[])
for a1 in 1:N_alg
    for a2 in (a1+1):N_alg
        push!(dfpred, (Algs[a2], Algs[a1], a2, a1))
        push!(dfpred, (Algs[a1], Algs[a2], a1, a2))
    end
end
Npreds = nrow(dfpred)
Y = Union{Number, Missing}[missing for _ in 1:Npreds]

preds = predict(m1(dfpred.A1, dfpred.A2, Y), chains; include_all = true)
Nparams = 1+N_alg # sigma and the 5 alphas
ss = summarystats(preds)
dfpred[!, :Prob] = round.(ss[(Nparams+1):end, :mean], digits=2)
dfpred[!, :Std] = round.(ss[(Nparams+1):end, :std], digits=2)
sort!(dfpred, [:Prob], rev=true)
select(dfpred, [:Alg2, :Alg1, :Prob, :Std])
dfpred[dfpred.Prob .>= 0.50, :]

# PSIS-LOO check
using ParetoSmooth
m1_psis_loo = psis_loo(m1(alg1, alg2, twowins), chains)
#Results of PSIS-LOO-CV with 10000 Monte Carlo samples and 200 data points. Total Monte Carlo SE of 0.018.
#┌───────────┬────────┬──────────┬───────┬─────────┐
#│           │  total │ se_total │  mean │ se_mean │
#├───────────┼────────┼──────────┼───────┼─────────┤
#│   cv_elpd │ -55.33 │     6.70 │ -0.28 │    0.03 │
#│ naive_lpd │ -52.41 │     6.21 │ -0.26 │    0.03 │
#│     p_eff │   2.92 │     0.51 │  0.01 │    0.00 │
#└───────────┴────────┴──────────┴───────┴─────────┘