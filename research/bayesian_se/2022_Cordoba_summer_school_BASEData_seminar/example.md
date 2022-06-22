# Bayesian Analysis Example: Which algorithm is best?

This is a small/simple example of how to perform a Bayesian Analysis of some software engineering related data. Since this was first presented at the International Summer School on Search- and Machine Learning-based Software Engineering the focus is on the question:

**RQ. Which search/optimization algorithm performs best?**

In particular, we used the [Julia](https://julialang.org/) library [`BlackBoxOptim`](https://github.com/robertfeldt/BlackBoxOptim.jl/) (version 0.6.1) to compare how five (5) different optimization algorithms performed on each of five (5) different optimization problems. 

These problems are synthetic benchmark problems for which we can vary the dimension of the problem, i.e. how many decision variables are to be optimized. This makes for a more realistic comparison since we don't simply run them for a fixed, pre-chosen number of dimensions. Instead we sampled dimensions in the range from 2 to 300. We also used a fixed optimization time for each run: 1 second plus 0.2 seconds per dimension. This is realistic since we know that problems are harder to optimize the larger dimension they have; we thus need to give the algorithms more time.

With 5 algorithms, 5 problems, 300 dimensions to try, running all combinations would lead to 5*5*300=7500 possible executions. Furthermore, the algorithms we evaluate are stochastic so can perform differently for each optimization run. Even if we would only run, say, 3 runs per setting we would now need to run more than 20 thousand optimization runs taking an average of 1+150*0.1=16 seconds, more than 5 days.

The specific problem we investigate is therefore:

**RQ. Which of 5 search/optimization algorithm from BlackBoxOptim performs best on 5 different problems up to a dimension of 300?**

Our key idea is that we want to reduce the execution effort by making only pair-wise comparisons of two randomly sampled algorithms, on one randomly selected problem, and with a randomly selected dimension in the range from 2 to 300. Given as few such pair-wise runs as possible, we want to use a statistical analysis that still enables us to answer the question, given as few runs as possible.

## Dataset

Pair-wise comparisons so each row of data has:

1. **Problem** - name of problem
2. **Dim** - dimension of problem used
3. **Time** - maximum time allowed for each optimization run
4. **Alg1** - the first algorithm run on this specific problem at this specific time
5. **Alg2** - the second algorithm run on this specific problem at this specific time
6. **Winner** - 0 if Alg1 performed the best, 1 if Alg2 performed the best. Best here means resulting in a lower optimum found by the algorithm (since all investigated problems are minimization problems).
7. **F1** - best (lowest) fitness found by Alg1
8. **F2** - best (lowest) fitness found by Alg2

I let this run during the night on my laptop and got a total of 1300 samples/entries/rows, although we will not use them all in all analyses.

## Bayesian modeling

For simplicity, and since there was limited time during the seminar, we implemented two different statistical models:

1. [M1](model1_bradley_terry.jl), the Bradley-Terry model with a single logit value per algorithm
2. [M2](model2_bradley_terry_varying_slope.jl), a more complex model that also takes the dimension of the search problem into account by modeling how the performance of the algorithms change with varying problem dimension.