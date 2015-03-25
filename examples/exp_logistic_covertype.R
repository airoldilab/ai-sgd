#!/usr/bin/env Rscript
# Compare optimization methods for logistic regression to predict forest cover
# type.
# UCI Repository: https://archive.ics.uci.edu/ml/datasets/Covertype
#
# Dimensions:
#   n=581,012 observations
#   p=54 parameters
#
# @pre Current working directory is the root directory of this repository
# @pre The following files exist and are downloaded from the above link.
#   "examples/data/covtype.data"

library(ggplot2)

source("functions.R")
source("functions_logistic.R")
source("sgd.R")

set.seed(42)
raw <- read.table("examples/data/covtype.data", sep=",")
#idxs <- sample(1:nrow(raw), floor(0.75*nrow(raw)))
idxs <- sample(1:nrow(raw), floor(0.01*nrow(raw))) # using very small training set
raw.train <- raw[idxs, ]
raw.test <- raw[-idxs, ]

# Form DATA object for training.
data.train <- list()
data.train$X <- as.matrix(raw.train[, -55])
data.train$Y <- raw.train[, 55]
data.train$model <- get.glm.model("logistic")

# Form DATA object for testing.
data.test <- list()
data.test$X <- as.matrix(raw.test[, -55])
data.test$Y <- raw.test[, 55]
data.test$model <- get.glm.model("logistic")

# Build models using training data and output error on test data.
sgd.methods <- c("AI-SGD", "SVRG", "SGD", "SGD", "SGD")
pars <- c(0.025, 0.025, 0.005, 0.0025, 0.001)
out <- run.logistic(data.train, data.test, sgd.methods, pars, npass=6)
print(out)
# TODO Not sure what's going on with SVRG here. Need to double check.
