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

source("functions.R")
source("sgd.R")

# Read in training data.
#train <- read.table("examples/data/mnist/train.csv", header=TRUE, sep=",")
# Read in subset of data, obtained by sampling 1e4 rows from original training
# data (set.seed(42); sample(1:42e3, size=1e4)).
train <- read.csv("examples/data/mnist_n4.csv")

# Construct learning rate. A tweak of the one in Ruppert.
lr <- function(n) {
  D <- 1/1e-2 # should roughly be 1/minimal eigenvalue of observed Fisher
  alpha <- 1
  return(D/(D+n)^alpha)
}

# Form DATA object to feed to stochastic gradient method.
data <- list()
data$X <- as.matrix(subset(train, select=-label))
data$model <- get.glm.model("logistic")

# Build 9 binary classifiers for multinomial logistic regression.
list.theta <- list()
for (i in 0:8) {
  # Build regression to predict "i" or "not i".
  data$Y <- rep(NA, nrow(train))
  data$Y[train$label != i] <- 0
  data$Y[train$label == i] <- 1
  # Run IA-SGD.
  list.theta[[i+1]] <- sgd(data, sgd.method="implicit", averaged=T, lr=lr)
}

# Examine training error for each binary classifier.
for (i in 0:8) {
  # Get last theta estimate for the ith binary classifier.
  theta.i <- matrix(list.theta[[i+1]][, ncol(list.theta[[i+1]])])
  # Predict yes or no based on cutoff of 0.5.
  y.pred <- logistic(data$X %*% theta.i)
  y.pred[y.pred <= 0.5] <- 0
  y.pred[y.pred > 0.5] <- 1
  # Rebuild true labels.
  y.true <- rep(NA, nrow(train))
  y.true[train$label != i] <- 0
  y.true[train$label == i] <- 1
  # Print training error.
  print(sprintf("Training Error on Binary Classifier on %i: %0.5f", i, sum(y.pred !=
    y.true)/length(y.pred)))
}

predict <- function(X, list.theta) {
  # Build a multiclassifier using the 9 logistic models.
  #
  # Args:
  #   X: design matrix
  #   list.theta: List of 9 parameter matrices corresponding to each logistic
  #               model.
  #
  # Returns:
  #   A nrow(X) x 10 matrix, where the (i,j)th cell is the probability the ith
  #   observation is the (j-1)th digit.
  stopifnot(class(X) == "matrix")
  preds <- matrix(NA, nrow=nrow(X), ncol=10)
  for (k in 1:9) {
    theta.k <- matrix(list.theta[[k]][, ncol(list.theta[[k]])])
    preds[, k] <- exp(X %*% theta.k)
  }
  preds[, 10] <- 1
  # Normalize values, and truncate probabilities which are roughly zero up to
  # precision.
  preds <- preds/rowSums(preds)
  preds[preds < 1e-8] <- 0
  return(preds)
}

# Build probability matrix, then use the maximum probability as the classifier.
train.pred.matrix <- predict(data$X, list.theta)
# note: subtract 1 since labels range from 0-9
train.pred <- unlist(apply(train.pred.matrix, 1, function(x) which.max(x))) - 1
print(sprintf("Training Error for Multinomial Logistic Regression: %0.5f",
  sum(train.pred != train$label)/nrow(train)))
