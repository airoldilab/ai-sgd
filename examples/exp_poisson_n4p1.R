#!/usr/bin/env Rscript
# Compare optimization methods for Poisson regression on simulated data from a
# Poisson distribution.
#
# Data generating process:
#   Y ~ Poisson(lambda), where lambda = exp(X %*% theta)
#     X ~ Normal(0, A), where A is a randomly generated matrix with
#       eigenvalues (0.01,...,0.01)
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

run.all <- function(dim.n=1e4, dim.p=1e1, sgd.alpha=100) {
  # Runs all experiments for the Xu setup.
  set.seed(42)
  A <- generate.A(dim.p)
  d <- sample.data(dim.n, A, glm.model = get.glm.model("poisson"),
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
  print("Running explicit SGD..")
  theta$SGD <- sgd(d, sgd.method="explicit",
                   lr=lr.explicit, alpha=sgd.alpha)
  print("Running implicit SGD..")
  theta$ISGD <- sgd(d, sgd.method="implicit",
                    lr=lr.implicit, alpha=sgd.alpha)
  print("Running averaged implicit SGD..")
  theta$`AI-SGD` <- sgd(d, sgd.method="implicit", averaged=T,
                        lr=lr.implicit, alpha=sgd.alpha)
  print("Running batch method..")
  theta$Batch <- batch(d, sequence=round(10^seq(
    log(dim.p + 10, base=10),
    log(dim.n, base=10), length.out=100))
    ) # the sequence is equally spaced points on the log scale

  # Plot and save image.
  #png("img/exp_poisson_n4p1.png", width=1280, height=720)
  plot.risk(d, theta, dim.n)
  #dev.off()
}
