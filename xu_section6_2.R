#!/usr/bin/env Rscript
# Replicate the experiment set in Xu, section 6.2, and compare other methods.
#
# @pre "img/" directory exists

library(dplyr)
library(ggplot2)
library(mvtnorm)

source("functions.R")
source("sgd.R")

batch <- function(data, sequence) {
  # Find the optimal parameter values using the batch method for a linear
  # model.
  #
  # Args:
  #   data: List of x, y, A, theta in particular form
  #   seq: Vector of iterate indices to calculate the batch method for. (In
  #        case you would only like to compute a subset of them.)
  #
  # Returns:
  #   p x niters matrix where the jth column is the jth theta update

  # check.data(data)
  n <- nrow(data$X)
  p <- ncol(data$X)
  # matrix of estimates of batch (p x niters)
  theta.batch <- matrix(0, nrow=p, ncol=length(sequence))
  colnames(theta.batch) <- sequence

  idx <- 1
  for (i in sequence) {
    if (i == 1) {
      dat <- as.data.frame(cbind(
        t(data$X[1:i, ]),
        data$Y[1:i]
        ))
    } else {
      dat <- as.data.frame(cbind(
        data$X[1:i, ],
        data$Y[1:i]
        ))
    }
    names(dat)[p+1] <- "Y"
    theta.batch[, idx] <- lm(Y ~ . + 0, data=dat)$coefficients
    idx <- idx + 1
  }

  return(theta.batch)
}

run.all <- function() {
  set.seed(42)
  A <- generate.A(p=100)
  d <- sample.data(dim.n=1e5, A)
  
  # Construct functions for learning rate.
  lr.explicit <- function(n, p, alpha) {
    gamma0 <- 1 / (sum(seq(0.01, 1, length.out=p)))
    alpha/(alpha/gamma0 + n)
  }
  lr.implicit <- function(n, alpha) {
    alpha/(alpha + n)
  }
  
  # Optimize!
  theta <- list()
  theta$sgd <- sgd(d, method="explicit", lr=lr.explicit, alpha=100)
  theta$asgd <- sgd(d, method="explicit", averaged=T, lr=lr.explicit, alpha=100)
  theta$isgd <- sgd(d, method="implicit", lr=lr.implicit, alpha=100)
  theta$batch <- batch(d, sequence=round(seq(1e2+1, 1e5, length.out=100)))
  
  # Reproduce the plot in Xu Section 6.2 and export it.
  png("img/xu_section6_2.png", width=1280, height=720)
  plot.risk(d, theta)
  dev.off()
  
}