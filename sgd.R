# An implementation of stochastic gradient methods for GLMs.

sgd <- function(data, sgd.method, lr, npass=1, lambda=0, ...) {
  # Find the optimal parameters using a stochastic gradient method for
  # generalized linear models.
  #
  # Args:
  #   data: DATA object created through sample.data(..) (see functions.R)
  #   sgd.method: a string which is one of the following: "SGD", "ASGD",
  #               "LS-SGD", "ISGD", "AI-SGD", "LS-ISGD", "SVRG"
  #   lr: function which computes learning rate with input the iterate index
  #   npass: number of passes over data
  #   lambda: L2 regularization parameter for cross validation. Defaults to
  #           performing no cross validation
  #
  # Returns:
  #   A p x n*npass+1 matrix where the jth column is the jth theta update.

  # Check input.
  stopifnot(
    all(is.element(c("X", "Y", "model"), names(data))),
    sgd.method %in% c("SGD", "ASGD", "LS-SGD", "ISGD", "AI-SGD", "LS-ISGD",
      "SVRG")
  )
  # Initialize constants.
  n <- nrow(data$X)
  p <- ncol(data$X)
  niters <- n*npass
  glm.model <- data$model
  # Initialize frequency and change niters if method is SVRG, following notation
  # in Johnson and Zhang (2013).
  m <- NULL
  if (sgd.method == "SVRG") {
    stopifnot(npass %% 2 == 0)
    m <- 2*n
    niters <- npass/2 # do this many 2-passes over the data
  }
  # Initialize parameter matrix for the stochastic gradient descent (p x n*npass+1).
  # Will return this matrix.
  theta.sgd <- matrix(0, nrow=p, ncol=niters+1)
  if (sgd.method == "SVRG") {
    # Mark the true number of iterations for each sgd iterate in SVRG (0, m, 2*m, ...).
    colnames(theta.sgd) <- 0:niters * m + 1
  }
  theta.new <- NULL
  ai <- NULL

  # Run the stochastic gradient method.
  # Main iteration: i = #iteration
  for (i in 1:niters) {
    # Index.
    idx <- ifelse(i %% n == 0, n, i %% n) # sample index of data
    xi <- data$X[idx, ]
    yi <- data$Y[idx]
    theta.old <- theta.sgd[, i]
    # Compute learning rate.
    ai <- lr(i, ...)

    # TODO: Test to see if the regularization actually works.
    if (sgd.method %in% c("SGD", "ASGD", "LS-SGD")) {
      theta.new <- sgd.update(theta.old, xi, yi, ai, lambda, glm.model)
    } else if (sgd.method %in% c("ISGD", "AI-SGD", "LS-ISGD")) {
      theta.new <- isgd.update(theta.old, xi, yi, ai, lambda, glm.model)
    } else if (sgd.method == "SVRG") {
      theta.new <- svrg.update(theta.old, data, lr, lambda, glm.model, m, ...)
    }
    theta.sgd[, i+1] <- theta.new
  }

  # Post-process parameters if the method requires it.
  if (sgd.method %in% c("ASGD", "AI-SGD")) {
    # Average over all estimates.
    theta.sgd <- average.post(theta.sgd)
  }
  if (sgd.method %in% c("LS-SGD", "LS-ISGD")) {
    # Run least squares fit over all estimates.
    theta.sgd <- ls.post(theta.sgd, data)
  }

  return(theta.sgd)
}

# Define update functions.
sgd.update <- function(theta.old, xi, yi, ai, lambda, glm.model) {
  # Shorthand for derivative of log-likelihood for GLMs with CV.
  score <- function(theta) {
    (yi - glm.model$h(sum(xi * theta))) * xi + lambda*sqrt(sum(theta^2))
  }
  theta.new <- theta.old + ai * score(theta.old)
  return(theta.new)
}
isgd.update <- function(theta.old, xi, yi, ai, lambda, glm.model) {
  # Make computation easier.
  xi.norm <- sum(xi^2)
  lpred <- sum(xi * theta.old)
  get.score.coeff <- function(ksi) {
    # Returns:
    #   The scalar value yi - h(theta_i' xi + xi^2 ξ) + λ*||theta_i+ξ||_2
    #TODO
    #yi - glm.model$h(lpred + xi.norm * ksi)
    yi - glm.model$h(lpred + xi.norm * ksi) + lambda*sqrt(sum((theta.old+ksi)^2))
  }
  # 1. Define the search interval.
  ri <- ai * get.score.coeff(0)
  Bi <- c(0, ri)
  if (ri < 0) {
    Bi <- c(ri, 0)
  }

  implicit.fn <- function(u) {
    u - ai * get.score.coeff(u)
  }
  # 2. Solve implicit equation.
  ksi.new <- NA
  if (Bi[2] != Bi[1]) {
    ksi.new <- uniroot(implicit.fn, interval=Bi)$root
  }
  else {
    ksi.new <- Bi[1]
  }
  theta.new <- theta.old + ksi.new * xi
  return(theta.new)
}
svrg.update <- function(theta.old, data, lr, lambda, glm.model, m, ...) {
  n <- nrow(data$X)
  p <- ncol(data$X)
  # Shorthand for derivative of log-likelihood for GLMs with CV.
  score <- function(theta) {
    (yi - glm.model$h(sum(xi * theta))) * xi + lambda*sqrt(sum(theta^2))
  }
  # Do one pass of data to obtain the average gradient.
  mu <- rep(0, p)
  for (idx in 1:n) {
    xi <- data$X[idx, ]
    yi <- data$Y[idx]
    mu <- mu + score(theta.old)/n
  }
  # Run inner loop, updating w by using a random sample.
  w <- theta.old
  for (mi in 1:m) {
    idx <- sample(1:n, 1)
    xi <- data$X[idx, ]
    yi <- data$Y[idx]
    ai <- lr(mi, ...)
    w <- w + ai * (score(w) - score(theta.old) + mu)
  }
  # Assign SGD iterate to the last updated weight ("option I").
  theta.new <- w
  return(theta.new)
}

# Define post-processing functions.
average.post <- function(theta.sgd) {
  return(t(apply(theta.sgd, 1, function(x) {
    cumsum(x)/(1:length(x))
  })))
}
ls.post <- function(theta.sgd, data) {
  # TODO: Generalize beyond Normal(0, A) data.
  n <- nrow(data$X)
  p <- ncol(data$X)
  ncol.theta <- ncol(theta.sgd) # n*npass+1
  # TODO: Generating y can be faster by doing matrix multiplication instead.
  # TODO: Benchmark this compared to forming y within the main SGD loop. The
  # latter method does not have to load in the DATA object into a function,
  # which is expensive.
  # Also the indices are probably off here: y[, 1] should not be all 0 (?).
  y <- matrix(0, nrow=p, ncol=ncol.theta)
  for (i in 1:(ncol.theta-1)) {
    idx <- ifelse(i %% n == 0, n, i %% n) # sample index of data
    xi <- data$X[idx, ]
    theta.old <- theta.sgd[, i]
    y[, i+1] <- data$obs.data$A %*% (xi - theta.old)
  }

  beta.0 <- matrix(0, nrow=p, ncol=ncol.theta)
  beta.1 <- matrix(0, nrow=p, ncol=ncol.theta)
  for (i in 1:ncol.theta) {
    x.i <- theta.sgd[, 1:i]
    y.i <- y[, 1:i]
    bar.x.i <- rowMeans(x.i)
    bar.y.i <- rowMeans(y.i)
    beta.1[, i] <- rowSums(y.i*(x.i - bar.x.i))/rowSums((x.i - bar.x.i)^2)
    beta.0[, i] <- bar.y.i - beta.1[, i] * bar.x.i
  }
  return(-beta.0/beta.1)
}
