using DataFrames, CSV

res = CSV.read(joinpath(@__DIR__(), "result_num_obs_sensitivity_analysis.csv"), DataFrame)

# Just plot with ggplot2 since we know it. Maybe clean up later so people
# need not install R and RCall etc...
using RCall
R"library(ggplot2)"

# Plot only for prior param 1.0, which is a good default
dfsel = res[res.PriorParam .== 1.0, :]
@rput dfsel
# Todo: Add error bars or indicate std in some other way. Nicer theme.
R"p <- ggplot(data=dfsel, aes(x=N, y=prob_ADE_vs_GSS)) + geom_line() + ylim(0.0, 1.0)"
R"ggsave('adaptive_de_vs_gss_over_varying_data_size.png', p)"

# Plot all 4 prior params in one graph
res.PriorStr = map(p -> "Exponential($p)", res.PriorParam)
@rput res
R"res$Prior <- as.factor(res$PriorStr)"
R"p <- ggplot(data=res, aes(x=N, y=prob_ADE_vs_GSS, color=Prior)) + geom_line() + ylim(0.0, 1.0)"
R"ggsave('adaptive_de_vs_gss_over_varying_data_size_and_priors.png', p)"

# We can see that the analysis is not sensitive to choice of prior and even though there
# is some early variation with few observations it seems stable from a range around 100-500
# data points.