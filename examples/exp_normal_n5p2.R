#!/usr/bin/env Rscript
# Compare optimization methods for linear regression on simulated data from a
# normal distribution.
#
# Data generating process:
#   Y = X %*% theta + epsilon, where
#     X ~ Normal(0, A), where A is a randomly generated matrix with eigenvalues
#     (1,1,1,0.02,0.02,...,0.02)
#     theta = (0,...,0)
#     epsilon ~ Normal(0,1)
# Dimensions:
#   n=1e5 observations
#   p=1e2 parameters
#
# @pre Current working directory is the root directory of this repository
# @pre Current working directory has the directory "img/"
# @pre Current working directory has the directory "out/"

library(MASS)
library(mvtnorm)

source("functions.R")
source("batch.R")
#source("sgd.R")

sgd <- function(data, sgd.method, averaged=F, ls=F, lr, ...) {
  # Find the optimal parameter values using a stochastic gradient method for a
  # linear model.
  #
  # Args:
  #   data: List of X, Y in particular form
  #   sgd.method: "explicit" or "implicit"
  #   averaged: boolean of whether or not to average estimates
  #   ls: boolean of whether or not to use least squares estimate
  #   lr: function which computes learning rate with input the iterate index
  #
  # Returns:
  #   p x (n+1) matrix where the jth column is the jth theta update

  # check.data(data)
  n <- nrow(data$X)
  p <- ncol(data$X)
  # Initialize parameter matrix for sgd (p x iters).
  theta.sgd <- matrix(0, nrow=p, ncol=n+1)
  # Initialize y matrix if least squares estimate desired (p x iters).
  if (ls == TRUE) y <- matrix(0, nrow=p, ncol=n+1)

  for (i in 1:n) {
    xi <- data$X[i, ]
    theta.old <- theta.sgd[, i]

    # Compute learning rate.
    if (sgd.method == "explicit") {
      ai <- lr(i, p, ...)
    } else if (sgd.method == "implicit") {
      ai <- lr(i, ...)
    }

    # Update.
    if (sgd.method == "explicit") {
      theta.new <- theta.old + (ai/2) * data$obs.data$A %*% (xi - theta.old)
    } else if (sgd.method == "implicit") {
      theta.new <- solve(diag(p) + (ai/2)*data$obs.data$A) %*% (theta.old +
      (ai/2)*data$obs.data$A%*%xi)
    }

    theta.sgd[, i+1] <- theta.new
  }

  if (averaged == TRUE) {
    # Average over all estimates.
    theta.sgd <- t(apply(theta.sgd, 1, function(x) {
      cumsum(x)/(1:length(x))
    }))
  }

  return(theta.sgd)
}

# Sample data.
set.seed(42)
X.list <- generate.X.A(n=1e5, p=1e2, lambdas=c(rep(1, 3), rep(0.02, 97)))
d <- generate.data(X.list, theta=matrix(0, ncol=1, nrow=1e2))

# Construct functions for learning rate according to Xu.
lr.explicit <- function(n, p) {
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

# Set method based on job.id.
job.id <- as.integer(commandArgs(trailingOnly=TRUE))
if (job.id == 1) {
  theta <- sgd(d, sgd.method="explicit", lr=lr.explicit)[, subset.idx]
} else if (job.id == 2) {
  theta <- sgd(d, sgd.method="implicit", lr=lr.implicit)[, subset.idx]
} else if (job.id == 3) {
  theta <- sgd(d, sgd.method="implicit", averaged=T, lr=lr.implicit)[, subset.idx]
} else if (job.id == 4) {
  theta <- batch(d, sequence=subset.idx, slope=F)
}

# Save outputs into individual files.
save(d, theta, file=sprintf("out/exp_normal_n5p2_%i.RData", job.id))
