#!/usr/bin/env Rscript
# Compare optimization methods for logistic regression to classify handwritten
# digits from MNIST data. Using Kaggle's split of training and test datasets.
# Kaggle: https://www.kaggle.com/c/digit-recognizer
#
# Dimensions:
#   n=42,000 observations
#   p=784 parameters
#
# @pre Current working directory is the root directory of this repository
# @pre Current working directory has the directory "out/"
# @pre The files "examples/data/mnist_train.csv" and
#      "examples/data/mnist_test.csv" exist and are downloaded from Kaggle.

library(ggplot2)

source("functions.R")
source("functions_logistic.R")
source("sgd.R")

set.seed(42)
raw <- read.csv("examples/data/mnist_train.csv")
#idxs <- sample(1:nrow(raw), floor(0.75*nrow(raw)))
idxs <- sample(1:nrow(raw), floor(0.01*nrow(raw))) # using very small training set
raw.train <- raw[idxs,]
raw.test <- raw[-idxs, ]

# Form DATA object for training.
data.train <- list()
data.train$X <- as.matrix(subset(raw.train, select=-label))
data.train$Y <- raw.train$label
data.train$model <- get.glm.model("logistic")

# Form DATA object for testing.
data.test <- list()
data.test$X <- as.matrix(subset(raw.test, select=-label))
data.test$Y <- raw.test$label
data.test$model <- get.glm.model("logistic")

# Build models using training data and output error on test data.
sgd.methods <- c("AI-SGD", "SVRG", "SGD", "SGD", "SGD")
pars <- c(0.025, 0.025, 0.005, 0.0025, 0.001)
out <- run.logistic(data.train, data.test, sgd.methods, pars, npass=10)
print(out)
