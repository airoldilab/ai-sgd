# Compare optimization methods for linear regression on simulated data from a
# normal distribution.
#
# Data generating process:
#   Y = X %*% θ + ɛ, where
#     X ~ Normal(0, A), where A is a randomly generated matrix with eigenvalues
#     (1,...,1)
#     θ = (1,...,1)
#     ɛ ~ Normal(0,1)
# Dimensions:
#   n=1e5 observations
#   p=1e1 parameters
#
# @pre Current working directory is the root directory of this repository
rm(list=ls())
library(MASS)
library(mvtnorm)

source("functions.R")
source("batch.R")
source("sgd.R")

# Sample data.
set.seed(42)
nsamples = 1e5 #n = #samples.
ncovs = 10  # p= #covariates
model = "gaussian"

X.list <- generate.X.A(n=nsamples, p=ncovs, lambdas=seq(1, 1, length.out=ncovs))
lambda0 = min(eigen(X.list$A)$values)
d <- generate.data(X.list, theta= rep(1, ncovs),
                   glm.model = get.glm.model(model))

# Construct functions for learning rate according to Xu.
lr.explicit <- function(n) {
  gamma0 = 1/ sum(diag(X.list$A))
  gamma0 / (1 + lambda0 * gamma0 * n)
}

lr.avg <- function(n) {
  gamma0 = 1/ sum(diag(X.list$A))
  gamma0 * (1 + lambda0 * gamma0 * n)^(-2/3)
}

lr.implicit <- function(n) {
  1 / (1 + lambda0 * n)
}

lr.implicit.avg <- function(n) {
  (0 + lambda0 * n)^(-0.8)
}

# Store only a subset of them.
subset.idx <- as.integer(seq(10, nsamples, length.out=50))
# c(seq(100, 900, by=100), seq(1000, 1e5, by=1000))
mse <- function(theta.t) {
  apply(theta.t, 2, function(col) {
    log(t(col-d$theta) %*% X.list$A %*% (col-d$theta))
  })
}

print("Running batch")
load(file="batch.rda")
theta.batch <- batch(d, sequence=subset.idx)
plot(mse(theta.batch), main="MSE normal", type="l", xlab="iteration (10^..)")
x.labels = sapply(subset.idx, function(i) sprintf("%.1f", log(i, base = 10)))
axis(1, at=1:length(subset.idx), labels=x.labels)
print(summary(mse(theta.batch)))
#
if(TRUE) {
  print("Running SGD.")
  theta.sgd <- sgd(d, sgd.method="SGD", lr=lr.explicit)[, subset.idx]
  lines(mse(theta.sgd), col="red")
  print(summary(mse(theta.sgd)))

  print("Running ASGD.")
  theta.asgd <- sgd(d, sgd.method="ASGD", lr=lr.avg)[, subset.idx]
  lines(mse(theta.asgd), col="pink")
  print(summary(mse(theta.asgd)))

  print("Running ISGD.")
  theta.isgd <- sgd(d, sgd.method="ISGD", lr=lr.implicit)[, subset.idx]
  lines(mse(theta.isgd), col="blue")
  print(summary(mse(theta.isgd)))
}

## Experiment for best AISGD
print("Running AI-SGD.")
theta.aisgd <- sgd(d, sgd.method="AI-SGD", lr=lr.implicit.avg)[, subset.idx]
lines(mse(theta.aisgd), col="cyan")
print(summary(mse(theta.aisgd)))
