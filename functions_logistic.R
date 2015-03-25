# This is a collection of functions used for building and testing multinomial
# logistic regressions.

train <- function(data, sgd.method, par, ...) {
  # Build k-1 binary classifiers for multinomial logistic regression, where k is
  # the number of categories.
  #
  # Args:
  #   data: DATA object created through generate.data(..) (see functions.R)
  #   sgd.method: options are documented in sgd(...)
  #   par: the constant learning rate
  #
  # Returns:
  #   A list of k-1 matrices, each of which corresponds to the theta matrix from
  #   running sgd(...) for each binary model.
  # Construct learning rate. A tweak of the one in Ruppert.
  lr <- function(n, par) {
    # Constant learning rate.
    par
  }
  list.theta <- list()
  categories <- sort(unique(data$Y))
  for (i in 1:(length(categories)-1)) {
    # Build regression to predict "i" or "not i".
    temp <- data$Y
    data$Y <- rep(0, length(data$Y))
    data$Y[temp == categories[i]] <- 1
    # Run AI-SGD.
    print(sprintf("Running %s, par %s for binary classifier on %i..", sgd.method,
      par, categories[i]))
    list.theta[[i]] <- sgd(data, sgd.method=sgd.method, lr=lr, par=par, ...)
    data$Y <- temp
    # Do this to make indexing the same format as SVRG's.
    if (sgd.method != "SVRG") {
      n <- nrow(data$X)
      npass <- (ncol(list.theta[[i]]) - 1)/n # since ncol(theta) = n*npass+1
      list.theta[[i]] <- list.theta[[i]][, (seq(0, npass, 2))*n + 1]
    }
  }
  return(list.theta)
}

test <- function(data, list.theta, npass, check.all=T) {
  # Output misclassification error of the data for the multiclassifier.
  #
  # Args:
  #   data: DATA object created through generate.data(..) (see functions.R)
  #   list.theta: List of k-1 parameter matrices corresponding to each binary
  #               model.
  #   npass: number of passes when training model
  #   check.all: boolean specifying whether to check error for theta after each
  #              pass or to check error only after the last pass
  #
  # Returns:
  #   Vector of error percentages from 0 to 1, where the ith element is the
  #   misclassification error after training over i-1 passes of the data.
  categories <- sort(unique(data$Y))
  if (check.all == TRUE) {
    idxs <- 1:ncol(list.theta[[1]])
  } else {
    idxs <- ncol(list.theta[[1]])
  }
  out <- rep(NA, length(idxs))
  for (i in 1:length(idxs)) {
    # Build probability matrix.
    y.pred.mat <- predict(data$X, list.theta, idx=idxs[i])
    # Take the arg max of the probabilities as the classification.
    y.pred.idx <- unlist(apply(y.pred.mat, 1, function(x) which.max(x)))
    y.pred <- categories[y.pred.idx]
    # Print error rate.
    out[i] <- sum(y.pred != data$Y)/length(data$Y)
  }
  return(out)
}

run.logistic <- function(data.train, data.test, sgd.methods, pars, npass=2, plt=F) {
  # Wrapper to optimize using each specified sgd.method with the corresponding
  # constant learning rate. Then obtain error rates and output plot or data
  # frame.
  #
  # Note: npass must be a multiple of 2! (for compatibility with comparisons to
  # SVRG)
  library(ggplot2)
  stopifnot(npass %% 2 == 0)
  dat <- as.data.frame(matrix(NA, ncol=3, nrow=(npass/2+1)*length(sgd.methods)))
  names(dat) <- c("method", "passes", "error")
  for (i in 1:length(sgd.methods)) {
    list.theta <- train(data.train, sgd.method=sgd.methods[i], par=pars[i],
      npass=npass)
    idx.range <- ((i-1)*(npass/2+1)+1):(i*(npass/2+1))
    dat[idx.range, "method"] <- sprintf("%s %s", sgd.methods[i], pars[i])
    dat[idx.range, "passes"] <- seq(0, npass, by=2)
    temp <- test(data.test, list.theta, npass)
    dat[idx.range, "error"] <- temp
  }
  if (plt == FALSE) {
    return(dat)
  } else {
    return(plt.logistic(dat))
  }
}

plt.logistic <- function(dat) {
  library(ggplot2)
  return(ggplot(dat, aes(x=passes, y=error, group=method, color=method)) +
    geom_line() +
    xlab("Number of passes") +
    ylab("Error rate") +
    ggtitle("Error rate over number of passes"))
}

predict <- function(X, list.theta, idx) {
  # Build a multiclassifier using the k-1 binary models.
  #
  # Args:
  #   X: design matrix
  #   list.theta: List of k-1 parameter matrices corresponding to each binary
  #               model.
  #   idx: the theta iterate to use.
  #
  # Returns:
  #   A nrow(X) x k matrix, where the (i,j)th cell is the probability the ith
  #   observation belongs to the kth category.
  stopifnot(class(X) == "matrix")
  preds <- matrix(NA, nrow=nrow(X), ncol=length(list.theta))
  for (k in 1:(length(list.theta)-1)) {
    theta.k <- matrix(list.theta[[k]][, idx])
    preds[, k] <- exp(X %*% theta.k)
  }
  preds[, length(list.theta)] <- 1
  # Normalize values, and truncate probabilities which are roughly zero up to
  # precision.
  preds <- preds/rowSums(preds)
  preds[preds < 1e-8] <- 0
  preds[is.nan(preds)] <- 1 # TODO: temporary bug fix, since these numbers are
                            # so large(?)
  return(preds)
}
