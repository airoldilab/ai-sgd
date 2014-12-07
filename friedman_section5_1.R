#!/usr/bin/env Rscript
# Replicate the experiment set in Friedman et. al, section 5.1, and compare
# other methods.

library(dplyr)
library(mvtnorm)
library(glmnet)
library(tidyr)

source("sgd.R")

# This function is taken directly from Panos' example.
genx2 = function(n,p,rho){
  #    generate x's multivariate normal with equal corr rho
  # Xi = b Z + Wi, and Z, Wi are independent normal.
  # Then Var(Xi) = b^2 + 1
  #  Cov(Xi, Xj) = b^2  and so cor(Xi, Xj) = b^2 / (1+b^2) = rho
  z=rnorm(n)
  if(abs(rho)<1){
    beta=sqrt(rho/(1-rho))
    x0=matrix(rnorm(n*p),ncol=p)
    A = matrix(z, nrow=n, ncol=p, byrow=F)
    x= beta * A + x0
  }
  if(abs(rho)==1){ x=matrix(z,nrow=n,ncol=p,byrow=F)}

  return(x)
}

# This function is taken directly from Panos' example.
sample.data <- function(dim.n, dim.p, rho=0.0, snr=1) {
  # Samples the dataset according to Friedman et. al.
  #
  # 1. sample covariates
  X = genx2(dim.n, dim.p, rho)
  # 2. ground truth params.
  theta = ((-1)^(1:dim.p))*exp(-2*((1:dim.p)-1)/20)

  f= X %*% theta
  e = rnorm(dim.n)
  k= sqrt(var(f)/(snr*var(e)))
  y=f+k*e
  return(list(y=y, X=X, theta=theta))
}

# This function is taken directly from Panos' example.
dist <- function(x, y) {
  if(length(x) != length(y))
    stop("MSE should compare vectors of same length")
  sqrt(mean((x-y)^2))
}

# Construct functions for learning rate.
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
  if(!exists("sgd")) {
    stop("sgd() does not exist. You need to source it.")
  }
  if(!exists("lr.explicit")) {
    stop("lr.explicit() does not exist. You need to construct it.")
  }
  if(!exists("lr.implicit")) {
    stop("lr.implicit() does not exist. You need to construct it.")
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
        dataset <- sample.data(dim.n=dim.n, dim.p=dim.p, rho=rho, snr=3.0)
        true.theta <- dataset$theta
        x <- dataset$X
        y <- dataset$y
        stopifnot(nrow(x) == dim.n, ncol(x) == dim.p)

        # Run glmnet (naive).
        new.dt <- system.time(
          {fit=glmnet(x, y, alpha=1, standardize=FALSE, type.gaussian="naive")}
        )[1]
        new.mse <- median(apply(fit$beta, 2, function(est) dist(est, true.theta)))
        timings <- rbind(timings, c(rho, i, new.dt, new.mse, "naive"))

        # Run glmnet (covariance).
        new.dt <- system.time(
          {fit=glmnet(x, y, alpha=1, standardize=FALSE, type.gaussian="covariance")}
        )[1]
        new.mse <- median(apply(fit$beta, 2, function(est) dist(est, true.theta)))
        timings <- rbind(timings, c(rho, i, new.dt, new.mse, "cov"))

        d <- list(X=dataset$X, Y=dataset$y)
        # Run stochastic gradient descent (explicit).
        new.dt <- system.time(
          {fit=sgd(d, method="explicit", lr=lr.explicit, rho=rho)}
        )[1]
        new.mse <- dist(fit[, ncol(fit)], true.theta)
        timings <- rbind(timings, c(rho, i, new.dt, new.mse, "explicit"))

        # Run stochastic gradient descent (implicit).
        new.dt <- system.time(
          {fit=sgd(d, method="implicit", lr=lr.implicit)}
        )[1]
        new.mse <- dist(fit[, ncol(fit)], true.theta)
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
