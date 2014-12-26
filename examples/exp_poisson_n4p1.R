#!/usr/bin/env Rscript
# Compare optimization methods for Poisson regression on simulated data from a
# Poisson distribution.
#
# Data generating process:
#   Y ~ Poisson(lambda), where lambda = exp(X %*% theta)
#     X ~ Normal(0, A), where A is a randomly generated matrix with
#       eigenvalues being equally spaced points from 0.01 to 1
#     theta = (2*exp(-1),...,2*exp(-1))
# Dimensions:
#   n=1e4 observations
#   p=1e1 parameters
#
# @pre Current working directory is the root directory of this repository
# @pre Current working directory has the directory "img/"

library(dplyr)
library(ggplot2)
library(mvtnorm)

source("functions.R")
source("batch.R")
source("sgd.R")

run("poisson", pars=c(10, -0.8), add.methods=c("SGD", "I-SGD", "Batch"))
