# Search heuristically for optimal learning rates for AI-SGD.
#
# @pre source("functions.R")
# @pre source("sgd.R")

tunePar <- function(data, lr) {
  # Tune parameter by using optim, comparing errors over a subset of the data.
  #
  # Args:
  #   data: DATA object created through sample.data(..) (see functions.R)
  #   lr: The specified learning rate for AI-SGD.
  #
  # Returns:
  #   The parameter achieving lowest error over the subset of data and the
  #   checked values.
  out <- optim(par=c(1,-1), fn=evalPar, method="Nelder-Mead", data=d, lr=lr)
  # Set back to true parameterization.
  out$par[1] <- exp(out$par[1])
  out$par[2] <- interval.map(0, 1, -1, -1/2, logistic(out$par[2]))
  return(out)
}
evalPar <- function(par, data, idx=1:min(1e3, nrow(data$X)), lr, param=T) {
  # Do a pass with AI-SGD using the fixed params to evaluate the error.
  #
  # Args:
  #   par: hyperparameters for the AI-SGD
  #   data: DATA object created through sample.data(..) (see functions.R)
  #   idx: Vector of indices to use as the subset of the data. Defaults to first
  #        1000.
  #   lr: The specified learning rate for AI-SGD.
  #   param: Whether to parametrize the parameters or not; set to F if testing
  #          values manually.
  #
  # Returns:
  #   The training error of AI-SGD using the fixed parameter values trained over
  #   the subset of the data.
  # Subset data.
  data$X <- data$X[idx, ]
  data$Y <- data$Y[idx]
  if (param == TRUE) {
    # Convert range from (-infty, infinity)^2 to [0, infty) x [-1, -1/2].
    par[1] <- exp(par[1])
    par[2] <- interval.map(0, 1, -1, -1/2, logistic(par[2]))
  }
  # Run SGD.
  theta.sgd <- sgd(data, sgd.method="AI-SGD", lr=lr, par=par)
  theta.sgd <- theta.sgd[, ncol(theta.sgd)]
  # Use mse of h(X*θ) from y.
  cost <- norm(data$Y - data$model$h(data$X %*% theta.sgd), type="2")
  if (length(par) == 1) {
    print(sprintf("Trying par=%0.3f yields cost %0.3f", par, cost))
  } else {
    print(sprintf("Trying par=(%s) yields cost %0.3f",
      paste(signif(par, digits=3), collapse=", "),
      cost))
  }
  return(cost)
}
