# Goal here is to do a prior sensitivity analysis, i.e. to what extent are our
# conclusions affected by the priors we start the analysis from?
# Main idea: Paramterize the prior so we can control it and then
# vary it and study the effect on conclusions.
include("common.jl")

# Bradley Terry model with a single logit value per algorithm
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
# Prepare data to be input in the model
#
#df = dffull # If want to run with all data
Nobs = nrow(df)
println("Using $Nobs observations")
const alg1 = Int[findfirst(==(a), Algs) for a in df.Alg1]
const alg2 = Int[findfirst(==(a), Algs) for a in df.Alg2]
const twowins = df.Winner

# Make a function that samples the posterior given the prior param(s)
function m1_posterior_for_prior(priorExpParam)
    m = m1_prior_parameterized(alg1, alg2, twowins, priorExpParam)
    chains = sample(m, DynamicNUTS(), 
        MCMCThreads(), 2_000, 5)
    return chains, m
end

c_1, m_1 = m1_posterior_for_prior(1.0)
c_05, m_05 = m1_posterior_for_prior(0.5)
c_01, m_01 = m1_posterior_for_prior(0.1)

using ParetoSmooth
ploo_1 = psis_loo(m_1, c_1)
ploo_05 = psis_loo(m_05, c_05)
ploo_01 = psis_loo(m_01, c_01)

models = (
        m_1=ploo_1,
        m_05=ploo_05,
        m_01=ploo_01
);
comps = loo_compare(models)

# m_1 is preferred but differences are very slight

# Check ranks summary stats:
summarize_ranks(calc_ranks(Algs, DataFrame(c_1), "a"), Algs)
summarize_ranks(calc_ranks(Algs, DataFrame(c_05), "a"), Algs)
summarize_ranks(calc_ranks(Algs, DataFrame(c_01), "a"), Algs)

# There are almost no differences so we are NOT sensitive to choice of prior
# if chosen in "normal" ranges.

# Not even with a very unusual prior do we get into problems, ranks are very close to unchanged:
c_100, m_100 = m1_posterior_for_prior(100.0)
summarize_ranks(calc_ranks(Algs, DataFrame(c_100), "a"), Algs)