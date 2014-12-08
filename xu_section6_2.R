#!/usr/bin/env Rscript
# Replicate the experiment set in Xu, section 6.2, and compare other methods.
#
# @pre "img/" directory exists
rm(list=ls())
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
  warning("Batch method needs to be updated to use data$model.")
  # TODO: Current implementation assumes linear normal model.
  #     Need to update using glm() and the appropriate family
  #     according to the specification in data$model$name.
  n <- nrow(data$X)
  p <- ncol(data$X)
  # matrix of estimates of batch (p x niters)
  theta.batch <- matrix(0, nrow=p, ncol=length(sequence))
  colnames(theta.batch) <- sequence

  idx <- 1
  # i = max datapoint to be included.
  stopifnot(length(sequence) > 0, sequence[1] > 1)
  for (i in sequence) {
    X = data$X[1:i, ]
    y = data$Y[1:i]
    theta.batch[, idx] <- as.numeric(lm(y ~ X + 0)$coefficients)
    idx <- idx + 1
  }
  
  return(theta.batch)
}

run.all <- function(dim.n=1e4, dim.p=1e1, sgd.alpha=100) {
  # Runs all experiments for the Xu setup.
  set.seed(42)
  A <- generate.A(dim.p)
  d <- sample.data(dim.n, A, glm.model = get.glm.model("gaussian"),
                   theta=2 * exp(-seq(1, dim.p)))
  
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
  cat("Running SGD explicit..\n")
  theta$sgd <- sgd(d, sgd.method="explicit", lr=lr.explicit, alpha=sgd.alpha)
  cat("Running averaged SGD explicit..\n")
  theta$asgd <- sgd(d, sgd.method="explicit", averaged=T, 
                    lr=lr.explicit, alpha=sgd.alpha)
  cat("Running SGD implicit..\n")
  theta$isgd <- sgd(d, sgd.method="implicit", 
                    lr=lr.implicit, alpha=sgd.alpha)
  
  if(d$model$name=="gaussian") {
    cat("Running batch method..\n")
    theta$batch <- batch(d, sequence=round(seq(dim.p + 10, dim.n, length.out=100)))  
  } else {
    warning("Silencing batch method -- needs update.")
  }
  
  # Reproduce the plot in Xu Section 6.2 and export it.
  # png("out/xu_section6_2.png", width=1280, height=720)
  plot.risk(d, theta, dim.n)
  # dev.off()
}