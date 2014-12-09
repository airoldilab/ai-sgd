# This is a collection of auxiliary functions for usage in other scripts.

################################################################################
# Generalized linear models
################################################################################

logistic <- function(x) {
  if(x > 20) return(1)
  if(x < -20) return(0)
  return(exp(x) / (1+exp(x)))
}

get.glm.model <- function(model="gaussian") {
  # Returns the link/link--deriv functions of the specified GLM model.
  if (model == "gaussian") {
    return(list(name=model,
                h=function(x) x,
                hprime=function(x) 1))
  } else if (model == "poisson") {
    return(list(name=model,
                h=function(x) exp(x),
                hprime=function(x) exp(x)))
  } else if (model == "logistic") {
    return(list(name=model,
                h=function(x) logistic(x),
                hprime=function(x) logistic(x) * (1-logistic(x))))
  } else {
    stop(sprintf("Model %s is not supported...", model))
  }
}

################################################################################
# Matrix generation
################################################################################

random.orthogonal <- function(p) {
  # Get an orthogonal matrix.
  B <- matrix(runif(p^2), nrow=p)
  qr.Q(qr(B))
}

random.matrix <- function(lambdas=seq(0.01, 1, length.out=100)) {
  # Generate a random matrix with the desired eigenvalues.
  #
  # Args:
  #   lambdas: vector of eigenvalues
  #
  # Returns:
  #   A p-by-p matrix with eigenvalues lambda.
  p <- length(lambdas)
  Q <- random.orthogonal(p)
  A <- Q %*% diag(lambdas) %*% t(Q)
  return(A)
}

generate.X.A <- function(n, p, lambdas=seq(0.01, 1, length.out=p)) {
  # Generate observations from Normal(0, A).
  #
  # Args:
  #   n: number of observations
  #   p: number of parameters
  #   lambdas: eigenvalues of A
  #
  # Returns:
  #   A list, where X is the design matrix and A is the covariance.
  library(mvtnorm)
  A <- random.matrix(lambdas)
  X <- rmvnorm(n, mean=rep(0, p), sigma=A)
  return(list(X=X, A=A))
}

generate.X.corr <- function(n, p, rho) {
  # Generate normal observations with equally correlated covariates.
  #
  # Args:
  #   n: number of observations per covariate
  #   p: number of covariates
  #   rho: correlation
  #
  # Returns:
  #   A list, where X is the n x p matrix and rho is the correlation.
  stopifnot(abs(rho) < 1)
  # Data generating process:
  # Xi = beta*Z + Wi, where Z, Wi ~ N(0, 1)
  #   Var(Xi) = beta^2 + 1
  #   Cov(Xi, Xj) = beta^2
  #   rho = cor(Xi, Xj) = beta^2 / (1+beta^2)
  Z <- rnorm(n, mean=0, sd=1)
  if (abs(rho) < 1) {
    beta <- sqrt(rho/(1-rho))
    W <- matrix(rnorm(n*p),ncol=p)
    Z.mat <- matrix(Z, nrow=n, ncol=p)
    X <- beta * Z.mat + W
  } else { # rho == 1
    X <- matrix(Z, nrow=n, ncol=p)
  }
  return(list(X=X, rho=rho))
}

################################################################################
# Data generation
################################################################################

generate.data <- function(X.list,
                          theta=matrix(1, ncol=1, nrow=ncol(X)),
                          glm.model=get.glm.model("gaussian"),
                          snr=Inf) {
  # Samples the dataset.
  #
  # Args:
  #   X.list: list whose element X is the design matrix, and whose other
  #           elements are any stored data used to generate X
  #   theta: true parameters
  #   glm.model: GLM model (see get.glm.model(..))
  #   snr: signal-to-noise ratio
  #
  # Returns:
  #   The DATA object, which is a list with the following elements:
  #     Y = outcomes (n x 1)
  #     X = covariates (n x p)
  #     theta = true params. (p x 1)
  #     L = X * theta
  #     model = GLM model (see get.glm.model(..))
  #     obs.data = any additional data used to generate X
  X <- X.list$X
  n <- nrow(X)
  p <- ncol(X)
  lpred <- X %*% theta
  # Generate outcomes according to the specified GLM.
  if (glm.model$name == "gaussian") {
    epsilon <- rnorm(n, mean=0, sd=1)
    k <- sqrt(var(lpred)/(snr*var(epsilon)))
    y <- lpred + k*epsilon
  } else if (glm.model$name == "poisson") {
    y <- rpois(n, lambda=glm.model$h(lpred))
  } else if (glm.model$name == "logistic") {
    y <- rbinom(n, size=1, prob=glm.model$h(lpred))
  } else {
    stop(sprintf("GLM model %s is not implemented..", glm.model$name))
  }
  # Store any additional data used to generate X (but not X itself again).
  X.list$X <- NULL
  # Return the DATA object.
  return(list(Y=y, X=X, theta=theta, L=lpred, model=glm.model, obs.data=X.list))
}

print.data <- function(data) {
  # Do a pretty print of the object generated from sample.data.
  nx <- nrow(data$X)
  ny <- length(data$Y)
  p <- ncol(data$X)
  stopifnot(nx==ny, p==length(data$theta))
  lambdas <- eigen(cov(data$X))$values
  print(lambdas)
  print(mean(data$Y))
  print(var(data$Y))
  print(1 + sum(cov(data$X)))
}

################################################################################
# Diagnostics
################################################################################

plot.risk <- function(data, est) {
  # Plot estimated biases of the optimization routines performed.
  # TODO: Generalize to not just Normal(0, A) models.
  #
  # Args:
  #   data: DATA object created through sample.data(..) (see functions.R)
  #   est: A list of matrix estimates, one for each optimization method run on
  #        data.
  #
  # Returns:
  #   A log-log scaled plot with a curve for each optimization routine,
  #   showing excess risk over training size.
  library(dplyr)
  library(ggplot2)

  list.bias <- list()
  # 1. iterate over method
  for (i in 1:length(est)) {
    # 1b. Get the risk values for specific method "i"
    values <- apply(est[[i]], 2, function(colum)
        t(colum-data$theta) %*% data$obs.data$A %*% (colum-data$theta))
    # 2. Get the risk (bias) values into a list.
    if (is.null(colnames(est[[i]]))) {
      list.bias[[i]] <- data.frame(
        t=1:length(values),
        est.bias=values,
        method=names(est)[i]
        )
    # This is to account for batch method, which we do not compute each
    # iteration but a subset of them.
    } else {
      list.bias[[i]] <- data.frame(
        t=as.integer(colnames(est[[i]])),
        est.bias=values,
        method=names(est)[i]
        )
    }
  }

  # Create a data frame row binding each p x niters matrix for ggplot.
  dat <- do.call(rbind, list.bias)

  # Plot.
  # Get range of iterations to plot (equivalent to c(dim.p, dim.n))
  iter.range <- c(max(sapply(est, nrow)), max(sapply(est, ncol)))
  # TODO: Make the plot a bit cleaner (e.g. larger size?)
  return(dat %>%
    ggplot(aes(x=t, y=est.bias, group=method, color=method)) +
      geom_line() +
      scale_x_log10(limits=iter.range, breaks=10^(0:9)) +
      scale_y_log10(limits=c(1e-4, 1e4), breaks=10^(seq(-6,6,2))) +
      xlab("Training size t") +
      ylab("Excess risk") +
      ggtitle("Excess risk over training size")
  )
}

################################################################################
# Miscellaneous
################################################################################

fracSec <- function() {
  # Generate a seed number based on the current time.
  now <- as.vector(as.POSIXct(Sys.time())) / 1000
  as.integer(abs(now - trunc(now)) * 10^8)
}
