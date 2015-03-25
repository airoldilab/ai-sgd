#!/usr/bin/env Rscript
# Compare optimization methods for linear regression on simulated data from a
# normal distribution.
#
# Data generating process:
#   Y = X %*% θ + ɛ, where
#     X ~ Normal(0, A), where A is a randomly generated matrix with eigenvalues
#     (1,1,1,0.02,0.02,...,0.02)
#     θ = (0,...,0)
#     ɛ ~ Normal(0,1)
# Dimensions:
#   n=1e5 observations
#   p=1e2 parameters
#
# @pre Current working directory is the root directory of this repository
# @pre Current working directory has the directory "img/"
# @pre Current working directory has the directory "out/"

library(dplyr)
library(ggplot2)
library(mvtnorm)

source("functions.R")
source("batch.R")
#source("sgd.R")

# sgd.R doesn't work because we assume no slope for this example
sgd.noslope <- function(data, sgd.method, lr, ...) {
  n <- nrow(data$X)
  p <- ncol(data$X)
  theta.sgd <- matrix(0, nrow=p, ncol=n+1)
  for(i in 1:n) {
    xi <- data$X[i, ]
    theta.old <- theta.sgd[, i]
    ai <- lr(i, ...)
    if (sgd.method %in% c("SGD", "ASGD")) {
      theta.new <- theta.old + (ai/2) * data$obs.data$A %*% (xi - theta.old)
    } else if (sgd.method %in% c("ISGD", "AI-SGD")) {
      theta.new <- solve(diag(p) + (ai/2)*data$obs.data$A) %*% (theta.old +
        (ai/2)*data$obs.data$A%*%xi)
    }
    theta.sgd[, i+1] <- theta.new
  }
  if (sgd.method %in% c("ASGD", "AI-SGD")) {
    theta.sgd <- t(apply(theta.sgd, 1, function(x) {
      cumsum(x)/(1:length(x))
    }))
  }
  return(theta.sgd)
}

# Sample data.
set.seed(42)
n <- 1e5
p <- 1e2
X.list <- generate.X.A(n, p, lambdas=c(rep(1, 3), rep(0.02, 97)))
d <- generate.data(X.list, theta=matrix(0, ncol=1, nrow=p))

# Construct functions for learning rate according to Xu.
lr.explicit <- function(n) {
  1/(1 + 0.02*n)
}
lr.avg <- function(n) {
  (1 + 0.02*n)^(-2/3)
}
lr.implicit <- function(n) {
  1/(1 + 0.02*n)
}

# Store only a subset of them.
subset.idx <- c(seq(100, 900, by=100), seq(1000, 1e5, by=1000))

theta <- list()
theta[["SGD"]] <- sgd.noslope(d, sgd.method="SGD", lr=lr.explicit)[, subset.idx]
theta[["ISGD"]] <- sgd.noslope(d, sgd.method="ISGD", lr=lr.implicit)[, subset.idx]
theta[["ASGD"]] <- sgd.noslope(d, sgd.method="ASGD", lr=lr.avg)[, subset.idx]
theta[["AI-SGD"]] <- sgd.noslope(d, sgd.method="AI-SGD", lr=lr.implicit)[, subset.idx]
print("Note: This batch requires the slope=F parameter to be uncommented!")
theta[["Batch"]] <- batch(d, sequence=subset.idx, slope=F)
for (i in 1:length(theta)) {
  colnames(theta[[i]]) <- subset.idx
}

plot.risk(d, theta)
