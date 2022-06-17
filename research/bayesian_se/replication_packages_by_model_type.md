# BDA in SE Examples and Replication Packages by Model Type

To get started doing BDA in SE it is often good to look at practical examples. Often such information is available per paper but from a learning perspective it can be useful to see examples of specific modeling types. Below we link to replication packages based on model type and going from simpler models to more complex ones. Our notion of complexity is somewhat subjective though but the criteria considered are:

1. How many predictor variables / co-variates does the data have?
2. Linear or non-linear model?
3. Complexity of interactions?
4. Varying intercepts?
5. Varying slopes?
6. Correlation structure?

Of course, this is rather ad hoc and other criteria or orderings can be considered.

## All replication packages, sorted by number of data points

Straight list of replication packages, in rough chronological order and by type of data it works with:

1. [Survey data on developer stress](https://github.com/torkar/rise2flow)
    - N = 187, 86 vars: 26 Binary, 3 Categorical, 50 Categorical likert, 3 Ordinal Integer, 4 Integer
2. [Toy (human height)data and ISBSG metrics data](https://github.com/torkar/BDA_in_ESE)
3. [Affective states in technical debt](https://github.com/torkar/affective_states)
    - N = 200, 11 vars: 4 Categorical, 5 Ordinal Integer, 2 Integer
4. [Feature selection in Requirements Engineering](https://github.com/torkar/feature-selection-RBS)
    - N = 11110, 18 vars: 7 Categorical, 11 Continuous
5. [Programming language data](https://github.com/torkar/BDA-PL)
    - N = 5102488, 17 vars: 1 Binary, 9 Categorical, 5 Integer, 2 Mixed