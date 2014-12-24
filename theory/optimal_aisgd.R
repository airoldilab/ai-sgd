# Search heuristically for optimal learning rates for AI-SGD.
# TODO: generalize to tuning grid of values instead of only a 1d interval
# maybe i should use a builtin optim-like routine which maximizes the
# parameters; use a wrapper function which outputs error

source("functions.R")
source("sgd.R")

tunePar <- function(data, k=2, ...) {
  # Tune parameter by searching via a scale down/up with a constant, comparing
  # errors over a subset of the data. Works only on 1-dimensional parameter
  # spaces.
  #
  # Args:
  #   data: DATA object created through sample.data(..) (see functions.R)
  #   k: Constant > 1 to multiply, which determines the refinement of the
  #      search.
  #
  # Returns:
  #   The parameter achieving lowest error over the subset of data and the
  #   checked values.
  low <- 1
  low.cost <- evalPar(low, data, ...)
  high <- low*k
  high.cost <- evalPar(high, data, ...)
  niters <- 0L # counter for number of iterations
  if (low.cost < high.cost) {
    while (low.cost < high.cost) {
      niters <- niters + 1L
      high <- low
      high.cost <- low.cost
      low <- low/k
      low.cost <- evalPar(low, data, ...)
    }
    best <- high
    best.cost <- high.cost
  } else if (high.cost < low.cost) {
    while (high.cost < low.cost) {
      niters <- niters + 1L
      low <- high
      low.cost <- high.cost
      high <- high*k
      high.cost <- evalPar(high, data, ...)
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

evalPar <- function(par, data, idx=1:min(1e3, nrow(data$X)), lr) {
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
  lambda0 <- exp(par[1])
  D <- interval.map(0, 1, -1/2, -1, logistic(par[2]))
  (1 + lambda0 * n)^D
}
lr.1d <- function(n, par) {
  # Learning rate where par is in [0, infty)
  lambda0 <- par
  D <- -1
  (1 + lambda0 * n)^D
}
#lr <- function(n, par) {
#  # Ruppert's learning rate.
#  # Note:
#  # alpha / (alpha + n) = 1 / (1 + lambda0*n), where lambda0 = 1/alpha
#  D <- 1
#  alpha <- par
#  par/n^D
#}

# Sample data.
set.seed(2)
nsamples <- 1e5 #n = #samples.
ncovs <- 10  # p = #covariates
model <- "gaussian"
X.list <- generate.X.A(n=nsamples, p=ncovs, lambdas=seq(1, 1, length.out=ncovs))
lambda0 <- min(eigen(X.list$A)$values)
d <- generate.data(X.list, theta=rep(5, ncovs),
                   glm.model=get.glm.model(model))

# Optimize using custom function which works on only 1 dimension.
par <- tunePar(d, lr=lr.1d, k=1.01)

# Optimize using optim.
vals <- optim(par=c(1,-1), fn=evalPar, method="Nelder-Mead", data=d, lr=lr)
# Set back to true parameterization.
vals$par[1] <- exp(vals$par[1])
vals$par[2] <- interval.map(0, 1, -1/2, -1, logistic(par[2]))

print(par)
print(vals)
