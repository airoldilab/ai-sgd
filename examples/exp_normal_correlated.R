#!/usr/bin/env Rscript
# Compare optimization methods for linear regression on simulated data from
# a correlated normal distribution.
#
# Data generating process:
#   Y = sum_{j=1}^p X_j*beta_j + k*epsilon, where
#     X ~ Multivariate normal where each covariate Xj, Xj' has equal correlation
#       rho; rho ranges over (0,0.1,0.2,0.5,0.9,0.95) for each pair (n, p)
#     beta_j = (-1)^j exp(-2(j-1)/20)
#     epsilon ~ Normal(0,1)
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
  # Run all experiments for the setup in Friedman et al.
  # Initialize list of results, where each list element is a pair (N,p).
  list.results <- list()
  for (i in 1:length(N)) {
    n <- N[i]
    p <- P[i]
    # Collect the benchmark data frame for each rho into a single data frame.
    list.results[[i]] <- data.frame()
    for (rho in rhos) {
      print(sprintf("(n, p, rho): (%i, %i, %0.2f)", n, p, rho))
      dat <- benchmark(n, p, rho)
      list.results[[i]] <- rbind(list.results[[i]], cbind(dat, rho))
    }
    names(list.results)[i] <- sprintf("n=%i, p=%i", n, p)
  }
  return(list.results)
}
