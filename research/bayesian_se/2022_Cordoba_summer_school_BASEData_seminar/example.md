# Bayesian Analysis Example: Which algorithm is best?

This is a small/simple example of how to perform a Bayesian Analysis of some software engineering related data. Since this was first presented at the International Summer School on Search- and Machine Learning-based Software Engineering the focus is on the question:

**RQ. Which search/optimization algorithm performs best?**

In particular, we used the [Julia](https://julialang.org/) library [`BlackBoxOptim`](https://github.com/robertfeldt/BlackBoxOptim.jl/) (version 0.6.1) to compare how five (5) different optimization algorithms performed on each of five (5) different optimization problems.

## Dataset



## Bayesian modeling

For simplicity, and since there was limited time during the seminar, we implemented two different statistical models:

1. M1, the Bradley-Terry model with a single logit value per algorithm
2. M2, a more complex model that also takes the dimension of the search problem into account by modeling how the performance of the algorithms change with varying problem dimension.