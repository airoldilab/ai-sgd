# This is a collection of auxiliary functions for usage in other scripts.
logistic <- function(x) {
  if(x > 20) return(1)
  if(x < -20) return(0)
  return(exp(x) / (1+exp(x)))
}

get.glm.model <- function(model="gaussian") {
  # Returns the link/link--deriv functions of the specified GLM model.
  #
  if(model=="gaussian") return(list(name=model,
                                    h=function(x) x,
                                    hprime=function(x) 1))
  if(model=="poisson") return(list(name=model,
                                   h=function(x) exp(x),
                                   hprime=function(x) exp(x)))
  if(model=="logistic") return(list(name=model,
                                    h=function(x) logistic(x),
                                    hprime=function(x) logistic(x) * (1-logistic(x))))
  stop(sprintf("Model %s is not supported...", model))
}


fracSec <- function() {
  # Generate a seed number based on the current time.
  now <- as.vector(as.POSIXct(Sys.time())) / 1000
  as.integer(abs(now - trunc(now)) * 10^8)
}

random.orthogonal <- function(p) {
  # Get an orthogonal matrix.
  B <- matrix(runif(p^2), nrow=p)
  qr.Q(qr(B))
}

generate.A <- function(p, lambdas=seq(0.01, 1, length.out=p)) {
  # Generate a random matrix with the desired eigenvalues.
  #
  # Args:
  #   p: dimension of matrix
  #   lambdas: vector of eigenvalues of length p
  #
  # Returns:
  #   A p-by-p matrix with eigenvalues lambda.
  Q <- random.orthogonal(p)
  A <- Q %*% diag(lambdas) %*% t(Q)
  return(A)
}

sample.data <- function(dim.n, A,
                        theta=matrix(1, ncol=1, nrow=nrow(A)),
                        glm.model=get.glm.model("gaussian")) {
  # Samples the dataset.
  #
  # Args:
  #   dim.n: size of dataset (#no. samples)
  #   A: covariance matrix for generating multivariate normal samples
  #   theta: true parameter values
  #   noise.sd = standard deviation of the noise.
  #
  # Returns:
  #   A list with (Y, X, A, true theta).
  library(mvtnorm)
  dim.p <- nrow(A)
  # This call will make the appropriate checks on A.
  X <- rmvnorm(dim.n, mean=rep(0, dim.p), sigma=A)
  lpred = X %*% theta

  if(glm.model$name=="gaussian") {
    epsilon <- rnorm(dim.n, mean=0, sd=1)
    # Data generation
    y <- lpred  + epsilon
  } else if(glm.model$name=="poisson") {
    y = rpois(dim.n, lambda = glm.model$h(lpred))
  } else if(glm.model$name=="logistic") {
    y = rbinom(dim.n, size = 1, prob=glm.model$h(lpred))
  } else {
    stop(sprintf("GLM model %s is not implemented..", glm.model$name))
  }
  # Return the DATA object.
  # Y = outcomes (n x 1)
  # X = covariates (n x p)
  # theta = true params. (p x 1)
  # L = X * theta
  # model = GLM model (see get.glm.model(..))
  # A = covariance matrix for the X.
  return(list(Y=y, X=X, L=lpred, model=glm.model, A=A, theta=theta))
}

check.data <- function(data) {
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

plot.risk <- function(data, est, max.iter) {
  # Plot estimated biases of the optimization routines performed.
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
        t(colum-data$theta) %*% data$A %*% (colum-data$theta))
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
  # TODO: Make the plot a bit cleaner (e.g. larger size?)
  return(dat %>%
    ggplot(aes(x=t, y=est.bias, group=method, color=method)) +
      geom_line() +
      scale_x_log10(limits=c(1, max.iter), breaks=10^(2:5)) +
      scale_y_log10(limits=c(1e-4, 1e4), breaks=10^(seq(-4,4,2))) +
      xlab("Training size t") +
      ylab("Excess risk") +
      ggtitle("Excess risk over training size")
  )
}
