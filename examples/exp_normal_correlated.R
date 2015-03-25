#!/usr/bin/env Rscript
# Compare optimization methods for linear regression on simulated data from
# a correlated normal distribution.
#
# Data generating process:
#   Y = sum_{j=1}^p X_j*β_j + k*ɛ, where
#     X ~ Multivariate normal where each covariate Xj, Xj' has equal correlation
#       ρ; ρ ranges over (0,0.1,0.2,0.5,0.9,0.95) for each pair (n, p)
#     β_j = (-1)^j exp(-2(j-1)/20)
#     ɛ ~ Normal(0,1)
#     k = 3
# Dimensions:
#   n=1000, p=100
#   n=5000, p=100
#   n=100, p=1000
#   n=100, p=5000
#   n=100, p=20000
#   n=100, p=50000
#
# @pre Current working directory is the root directory of this repository

library(glmnet)

source("functions.R")
source("sgd.R")

benchmark.all <- function(N=c(1000, 5000, 100, 100, 100, 100),
                          P=c(100, 100, 1000, 5000, 20000, 50000),
                          rhos=c(0.0, 0.1, 0.2, 0.5, 0.9, 0.95)) {
  # By default, run all experiments for the setup in Friedman et al.
  # Collect the benchmark data, running over every combination.
  out <- data.frame()
  for (i in 1:(length(N)*length(rhos))) {
    if (i %% length(N) != 0) {
      n <- N[i %% length(N)]
      p <- P[i %% length(N)]
    } else {
      n <- N[length(N)]
      p <- P[length(P)]
    }
    rho <- rep(rhos, each=length(N))[i] # use the same rho for length(N) times
    print(sprintf("(n, p, rho): (%i, %i, %0.2f)", n, p, rho))
    out <- rbind(out, cbind(n, p, rho, benchmark(n, p, rho)))
  }
  names(out)[1:3] <- c("n", "p", "rho")
  return(out)
}
