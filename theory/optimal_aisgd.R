# Search heuristically for optimal learning rates for AI-SGD.
# TODO: generalize to tuning grid of values instead of only a 1d interval
# maybe i should use a builtin optim-like routine which maximizes the
# parameters; use a wrapper function which outputs error

source("functions.R")
source("sgd.R")

tunePar <- function(data, idx=1:min(1e3, nrow(data$X)), k=2) {
  # Tune parameter by searching via a scale down/up with a constant, comparing
  # errors over a subset of the data. Works only on 1-dimensional parameter
  # spaces.
  #
  # Args:
  #   data: DATA object created through sample.data(..) (see functions.R)
  #   idx: Vector of indices to use as the subset of the data. Defaults to first
  #        1000.
  #   k: Constant > 1 to multiply, which determines the refinement of the
  #      search.
  #
  # Returns:
  #   The parameter achieving lowest error over the subset of data and the
  #   checked values.
  low <- 1
  low.cost <- evalPar(low, data, idx)
  high <- low*k
  high.cost <- evalPar(high, data, idx)
  niters <- 0L # counter for number of iterations
  if (low.cost < high.cost) {
    while (low.cost < high.cost) {
      niters <- niters + 1L
      high <- low
      high.cost <- low.cost
      low <- low/k
      low.cost <- evalPar(low, data, idx)
    }
    best <- high
    best.cost <- high.cost
  } else if (high.cost < low.cost) {
    while (high.cost < low.cost) {
      niters <- niters + 1L
      low <- high
      low.cost <- high.cost
      high <- high*k
      high.cost <- evalPar(high, data, idx)
    }
    best <- low
    best.cost <- low.cost
  } else {
    warning(sprintf("Both par=%0.3f and %0.3f have same cost %0.3f", low,
      high, low.cost))
    best <- low
    best <- low.cost
  }
  print(sprintf("Best is par=%0.3f with cost %0.3f", best, best.cost))
  print(sprintf("Number of iterations: %i", niters))
  return(c(best, best.cost))
}

evalPar <- function(par, data, idx=1:min(1e3, nrow(data$X))) {
  # Do a pass with AI-SGD using the fixed params to evaluate the error.
  #
  # Args:
  #   data: DATA object created through sample.data(..) (see functions.R)
  #   idx: Vector of indices to use as the subset of the data. Defaults to first
  #        1000.
  #   par: hyperparameters for the AI-SGD
  #
  # Returns:
  #   The training error of AI-SGD using the fixed parameter values trained over
  #   the subset of the data.
  # Subset data.
  data$X <- data$X[idx, ]
  data$Y <- data$Y[idx]
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

lr <- function(n, par) {
  D <- -1/2
  lambda0 <- par
  (1 + lambda0 * n)^D
}

# Sample data.
set.seed(42)
nsamples <- 1e5 #n = #samples.
ncovs <- 10  # p = #covariates
model <- "gaussian"
X.list <- generate.X.A(n=nsamples, p=ncovs, lambdas=seq(1, 1, length.out=ncovs))
lambda0 <- min(eigen(X.list$A)$values)
d <- generate.data(X.list, theta=rep(5, ncovs),
                   glm.model=get.glm.model(model))
par <- tunePar(d, k=1.01)
