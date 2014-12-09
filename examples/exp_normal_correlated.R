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

library(dplyr)
library(mvtnorm)
library(glmnet)
library(tidyr)

source("functions.R")
source("sgd.R")

dist <- function(x, y) {
  if (length(x) != length(y)) {
    stop("MSE should compare vectors of same length")
  }
  sqrt(mean((x-y)^2))
}

# Construct learning rate functions.
lr.explicit <- function(n, p, rho) {
  b <- rho/(1-rho)
  gamma0 <- 1/((b^2+1)*p)
  lambda0 <- 1
  alpha <- 1/lambda0
  return(alpha/(alpha/gamma0 + n))
}
lr.implicit <- function(n) {
  lambda0 <- 1
  alpha <- 1/lambda0
  return(alpha/(alpha + n))
}

simul.test <- function(dim.n,
                       dim.p,
                       rho.values=c(0.0, 0.1, 0.2, 0.5, 0.9, 0.95),
                       nreps=3) {
  # Run naive glmnet, covariance glmnet, explicit stochastic gradient descent,
  # and implicit stochastic gradient descent for various parameter values.
  #
  # Args:
  #   Self-explanatory.
  #
  # Returns:
  #   A data frame displaying rho, the simulation number, the time it took,
  #   the mean squared error, and the method used.
  library(glmnet)
  for (fn in c("sgd", "generate.X.corr", "generate.data", "lr.explicit",
               "lr.implicit")) {
    if(!exists(fn)) stop(sprintf("%s does not exist.", fn))
  }

  # Initialize.
  niters <- 0
  timings <- matrix(nrow=0, ncol=5)
  colnames(timings) <- c("rho", "rep", "time", "mse", "method")

  total.iters <- nreps * length(rho.values)
  pb <- txtProgressBar(style=3)
  seeds <- sample(1:1e9, size=total.iters)

  for(i in 1:nreps) {
    for(rho in rho.values) {
        # Set seed.
        niters <- niters + 1
        set.seed(seeds[niters])

        # Sample data.
        X.list <- generate.X.corr(dim.n, dim.p, rho=rho)
        theta <- ((-1)^(1:dim.p))*exp(-2*((1:dim.p)-1)/20)
        dataset <- generate.data(X.list, theta, snr=3)
        X <- dataset$X
        y <- dataset$Y
        stopifnot(nrow(X) == dim.n, ncol(X) == dim.p)

        # Run glmnet (naive).
        new.dt <- system.time(
          {fit=glmnet(X, y, alpha=1, standardize=FALSE, type.gaussian="naive")}
        )[1]
        new.mse <- median(apply(fit$beta, 2, function(est) dist(est, theta)))
        timings <- rbind(timings, c(rho, i, new.dt, new.mse, "naive"))

        # Run glmnet (covariance).
        new.dt <- system.time(
          {fit=glmnet(X, y, alpha=1, standardize=FALSE, type.gaussian="covariance")}
        )[1]
        new.mse <- median(apply(fit$beta, 2, function(est) dist(est, theta)))
        timings <- rbind(timings, c(rho, i, new.dt, new.mse, "cov"))

        # Run stochastic gradient descent (explicit).
        new.dt <- system.time(
          {fit=sgd(dataset, sgd.method="explicit", lr=lr.explicit, rho=rho)}
        )[1]
        new.mse <- dist(fit[, ncol(fit)], theta)
        timings <- rbind(timings, c(rho, i, new.dt, new.mse, "explicit"))

        # Run stochastic gradient descent (implicit).
        new.dt <- system.time(
          {fit=sgd(dataset, sgd.method="implicit", lr=lr.implicit)}
        )[1]
        new.mse <- dist(fit[, ncol(fit)], theta)
        timings <- rbind(timings, c(rho, i, new.dt, new.mse, "implicit"))

       setTxtProgressBar(pb, niters/total.iters)
    }

  }
  timings <- as.data.frame(timings)
  timings$time <- as.numeric(as.character(timings$time))
  timings$mse <- as.numeric(as.character(timings$mse))
  print("") # newline
  return(timings)
}

# Simulate glmnet for each scenario of N and p.
N <- c(1000, 5000, 100, 100, 100, 100)
P <- c(100, 100, 1000, 5000, 20000, 50000)
list.times <- list()

for (i in 1:length(N)) {
  print(sprintf("Simulation test %i of %i", i, length(N)))
  dat <- simul.test(dim.n=N[i], dim.p=P[i]) %>%
    group_by(rho, method) %>%
    summarize(time=mean(time), mse=mean(mse)) %>%
    ungroup()
  list.times[[i]] <- list()
  names(list.times)[i] <- paste("N=", N[i], " p=", P[i], sep="")
  list.times[[i]]$time <- dat %>%
    select(-mse) %>%
    spread(rho, time) %>%
    arrange(method)
  list.times[[i]]$mse <- dat %>%
    select(-time) %>%
    spread(rho, mse) %>%
    arrange(method)
}

print(list.times, digits=3)
