#
# 1. Load data
#
using CSV, DataFrames

cd(@__DIR__())
df = CSV.read("bbo_5alg_5problem_pairwise_comparisons_full.csv", DataFrame)

const Problems = sort(unique(df.Problem))
const Algs = sort(unique(vcat(df.Alg1, df.Alg2)))
const N_prob = length(Problems) # we set the used problem to 1 and other to 0
const N_alg = length(Algs) # we set left alg to -1, right alg to +1, rest to 0

# Now split so first 200 is the default and later we can include rest
dffull = df
df = dffull[1:200, :]

#
# 2. Load packages we will use
#
using Turing, Distributions
using MCMCChains, Plots, StatsPlots
using StatsFuns: logistic
using MLDataUtils: shuffleobs, stratifiedobs, rescale!
using DynamicHMC

function calc_ranks(Algs, posterior, prefix = "a")
    colnames = map(i -> prefix * "[$i]", 1:length(Algs))
    ranks = zeros(Int, nrow(posterior), length(Algs))
    for i in 1:nrow(posterior)
        ranks[i, :] = sortperm(collect(posterior[i, colnames]), rev=true)
    end
    return ranks
end

function summarize_ranks(ranks, Algs)
    df = DataFrame(Algorithm = String[], 
        MedianRank = Float64[], MeanRank = Float64[], Std = Float64[], 
        Q2_5 = Float64[], Q97_5 = Float64[],
        Q25 = Float64[], Q75 = Float64[])
    r2(v) = round(v, digits=2)
    for i in sortperm(vec(median(ranks, dims=1)))
        rs = ranks[:, i]
        push!(df, (Algs[i], r2(median(rs)), r2(mean(rs)), r2(std(rs)), quantile(rs, [0.025, 0.975, 0.25, 0.75])...)) 
    end
    df
end