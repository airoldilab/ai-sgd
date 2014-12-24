# Search heuristically for optimal learning rates for AI-SGD.

source("functions.R")
source("sgd.R")

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
evalPar <- function(par, data, idx=1:min(1e3, nrow(data$X)), lr) {
  # Do a pass with AI-SGD using the fixed params to evaluate the error.
  #
  # Args:
  #   par: hyperparameters for the AI-SGD
  #   data: DATA object created through sample.data(..) (see functions.R)
  #   idx: Vector of indices to use as the subset of the data. Defaults to first
  #        1000.
  #   lr: The specified learning rate for AI-SGD.
  #
  # Returns:
  #   The training error of AI-SGD using the fixed parameter values trained over
  #   the subset of the data.
  # Subset data.
  data$X <- data$X[idx, ]
  data$Y <- data$Y[idx]
  # Convert range from (-infty, infinity)^2 to [0, infty) x [-1, -1/2].
  par[1] <- exp(par[1])
  par[2] <- interval.map(0, 1, -1, -1/2, logistic(par[2]))
  # Run SGD.
  theta.sgd <- sgd(data, sgd.method="implicit", averaged=T, lr=lr, par=par)
  theta.sgd <- theta.sgd[, ncol(theta.sgd)]
  # Use mse of h(X*theta) from y.
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

# Auxiliary functions.
logit <- function(x) {
  # Return logit.
  return(log(x/(1-x)))
}
interval.map <- function(a, b, c, d, x) {
  # Scale values in [a,b] to [c,d].
  return(c + (d-c)/(b-a) * (x-a))
}

# Construct functions for learning rate.
lr <- function(n, par) {
  # Learning rate where par is a pair of numbers in (-infty, infty)
  lambda0 <- par[1]
  D <- par[2]
  (1 + lambda0 * n)^D
}
#lr <- function(n, par) {
#  # Ruppert's learning rate.
#  # Note:
#  # alpha / (alpha + n) = 1 / (1 + lambda0*n), where lambda0 = 1/alpha
#  D <- par[1]
#  alpha <- par[2]
#  alpha*n^D
#}

# Generate data.
set.seed(42)
nsamples <- 1e5 #n = #samples.
ncovs <- 10  # p = #covariates
model <- "gaussian"
X.list <- generate.X.A(n=nsamples, p=ncovs, lambdas=seq(1, 1, length.out=ncovs))
lambda0 <- min(eigen(X.list$A)$values)
d <- generate.data(X.list, theta=rep(5, ncovs),
                   glm.model=get.glm.model(model))

vals <- tunePar(d, lr=lr)
