# This is a collection of auxiliary functions for usage in other scripts.

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

sample.data <- function(dim.n, A, theta=matrix(1, ncol=1, nrow=nrow(A)),
                        model="gaussian") {
  # Samples the dataset.
  #
  # Args:
  #   dim.n: size of dataset
  #   A: covariance matrix for generating multivariate normal samples
  #   theta: true parameter values
  #
  # Returns:
  #   A list with (Y, X, A, true theta).
  dim.p <- nrow(A)
  # This call will make the appropriate checks on A.
  X <- rmvnorm(dim.n, mean=rep(0, dim.p), sigma=A)
  epsilon <- rnorm(dim.n, mean=0, sd=1)
  # Data generation
  y <- X %*% theta  + epsilon
  return(list(Y=y, X=X, A=A, theta=theta))
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

plot.risk <- function(data, est) {
  # Plot estimated biases of the optimization routines performed.
  #
  # Args:
  #   data: List of x, y, A, theta in particular form
  #   est: A list of matrix estimates, one for each set of optimization
  #        methods done on the data.
  #
  # Returns:
  #   A log-log scaled plot with a curve for each optimization routine,
  #   showing excess risk over training size.
  library(dplyr)
  library(ggplot2)

  list.bias <- list()
  for (i in 1:length(est)) {
    values <- apply(est[[i]], 2, function(colum)
        t(colum-data$theta) %*% data$A %*% (colum-data$theta))
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
  return(dat %>%
    ggplot(aes(x=t, y=est.bias, group=method, color=method)) +
      geom_line() +
      scale_x_log10(limits=c(1e2, 1e5), breaks=10^(2:5)) +
      scale_y_log10(limits=c(1e-4, 1e4), breaks=10^(seq(-4,4,2))) +
      xlab("Training size t") +
      ylab("Excess risk") +
      ggtitle("Excess risk over training size")
  )
}
