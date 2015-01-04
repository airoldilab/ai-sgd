# Form plots displaying error of AI-SGD using various learning rates, one of
# which uses the parameters obtained from tuning.

library(dplyr)
library(ggplot2)
library(mvtnorm)

source("functions.R")
source("sgd.R")
source("theory/optimal_aisgd.R")

# Set this to TRUE in order to have the following code also tune the parameters
bool.tune <- FALSE

# Construct functions for learning rate.
lr <- function(n, par) {
  # Ruppert's learning rate.
  # Note:
  # α / (α + n) = 1 / (1 + lambda0*n), where lambda0 = 1/α
  D <- par[1]
  alpha <- par[2]
  D*n^alpha
}

################################################################################
# Normal, n=1e5, p=1e2
################################################################################
if (bool.tune) {
  # Generate data.
  set.seed(42)
  n <- 1e5
  p <- 1e2
  model <- "gaussian"
  X.list <- generate.X.A(n, p)
  d <- generate.data(X.list, theta=rep(5, p),
                     glm.model=get.glm.model(model))
  # Tune learning rate parameters.
  vals <- tunePar(d, lr=lr)
}

# Optimal is (apparently) (Infty, -1/2).
#pars <- rbind(c(1e6, -1/2), c(1e1, -1), c(1/0.01, -1))
# Testing constant learning rates.
pars <- rbind(c(0.001,0), c(0.005, 0))
run("gaussian", pars=pars, n=1e4, p=1e2, add.methods="SGD")

################################################################################
# Poisson, n=1e4, p=10
################################################################################
if (bool.tune) {
  # Generate data.
  set.seed(42)
  n <- 1e4
  p <- 1e1
  X.list <- generate.X.A(n, p)
  model <- "poisson"
  d <- generate.data(X.list,
                     glm.model=get.glm.model(model),
                     theta=2 * exp(-seq(1, p)))
  # Tune learning rate parameters.
  vals <- tunePar(d, lr=lr)
}

# Optimal is roughly (10, -0.8637).
pars <- rbind(c(10, -0.8), c(1/0.01, -1))
run("poisson", pars=pars, n=1e4, p=1e1)

################################################################################
# Logistic, n=1e4, p=1e2
################################################################################
if (bool.tune) {
  # Generate data.
  set.seed(42)
  n <- 1e4
  p <- 1e2
  X.list <- generate.X.A(n, p)
  model <- "logistic"
  d <- generate.data(X.list,
                     glm.model=get.glm.model(model),
                     theta=2 * exp(-seq(1, p)))
  # Tune learning rate parameters.
  vals <- tunePar(d, lr=lr)
}

# Optimal is roughly (0.892, -0.5).
pars <- rbind(c(0.892, -0.5), c(1/0.01, -1))
run("logistic", pars=pars, n=1e4, p=1e2)
