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
  # Initialize parameter matrix for the stochastic gradient descent (p x n*npass+1).
  # Will return this matrix.
  theta.sgd <- matrix(0, nrow=p, ncol=niters+1)
  theta.new <- NULL
  ai <- NULL
  # Initialize frequency and change niters if method is SVRG, following notation
  # in Johnson and Zhang (2013).
  m <- NULL
  if (sgd.method == "SVRG") {
    stopifnot(npass %% 2 == 0)
    m <- 2*n
    niters <- npass/2 # do this many 2-passes over the data
  }

  # Run the stochastic gradient method.
  # Main iteration: i = #iteration
  # TODO: Test to see if the regularization actually works.
  for (i in 1:niters) {
    if (sgd.method %in% c("SGD", "ASGD", "LS-SGD")) {
      theta.new <- sgd.update(i, data, theta.sgd, lr, lambda, ...)
    } else if (sgd.method %in% c("ISGD", "AI-SGD", "LS-ISGD")) {
      theta.new <- isgd.update(i, data, theta.sgd, lr, lambda, ...)
    } else if (sgd.method == "SVRG") {
      theta.new <- svrg.update(i, data, theta.sgd, lr, lambda, ...)
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
sgd.update <- function(i, data, theta.sgd, lr, lambda, ...) {
  n <- nrow(data$X)
  glm.model <- data$model
  # Index.
  idx <- ifelse(i %% n == 0, n, i %% n) # sample index of data
  xi <- data$X[idx, ]
  yi <- data$Y[idx]
  theta.old <- theta.sgd[, i]
  # Calculate learning rate.
  ai <- lr(i, ...)
  # Shorthand for derivative of log-likelihood for GLMs with CV.
  score <- function(theta) {
    #TODO
    #(yi - glm.model$h(sum(xi * theta))) * xi
    (yi - glm.model$h(sum(xi * theta))) * xi + lambda*sqrt(sum(theta^2))
  }
  theta.new <- theta.old + ai * score(theta.old)
  return(theta.new)
}
isgd.update <- function(i, data, theta.sgd, lr, lambda, ...) {
  n <- nrow(data$X)
  glm.model <- data$model
  # Index.
  idx <- ifelse(i %% n == 0, n, i %% n) # sample index of data
  xi <- data$X[idx, ]
  yi <- data$Y[idx]
  theta.old <- theta.sgd[, i]
  # Make computation easier.
  xi.norm <- sum(xi^2)
  lpred <- sum(xi * theta.old)
  y.pred <- glm.model$h(lpred)  # link function of GLM
  # Calculate learning rate.
  ai <- lr(i, ...)
  # 1. Define the search interval.
  #TODO
  #ri <- ai * (yi - y.pred)
  ri <- ai * ((yi - y.pred) + lambda*sqrt(sum(theta.old^2)))
  Bi <- c(0, ri)
  if (ri < 0) {
    Bi <- c(ri, 0)
  }

  implicit.fn <- function(u) {
    #TODO
    #u - ai * (yi - glm.model$h(lpred + xi.norm * u))
    #u - ai * ((yi - glm.model$h(lpred + xi.norm * u)) + lambda*sqrt(sum(theta.old+u^2)))
    u - ai * ((yi - glm.model$h(lpred + xi.norm * u)) + lambda*sqrt(sum(u^2)))
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
  #TODO
  #theta.new <- theta.old + ksi.new
  return(theta.new)
}
svrg.update <- function(i, data, theta.sgd, lr, lambda, ...) {
  n <- nrow(data$X)
  glm.model <- data$model
  # Index.
  idx <- ifelse(i %% n == 0, n, i %% n) # sample index of data
  xi <- data$X[idx, ]
  yi <- data$Y[idx]
  theta.old <- theta.sgd[, i]
  # Calculate learning rate.
  ai <- lr(i, ...)
  # Shorthand for derivative of log-likelihood for GLMs with CV.
  score <- function(theta) {
    #TODO
    #(yi - glm.model$h(sum(xi * theta))) * xi
    (yi - glm.model$h(sum(xi * theta))) * xi + lambda*sqrt(sum(theta^2))
  }
  # Do one pass of data to obtain the average gradient.
  mu <- 0
  for (idx in 1:n) {
    xi <- data$X[idx, ]
    yi <- data$Y[idx]
    lpred <- sum(xi * theta.old)
    y.pred <- glm.model$h(lpred)  # link function of GLM
    mu <- mu + score(theta.old)
  }
  mu <- 1/n * mu
  # Update w by using a random sample.
  w <- c(theta.old, rep(NA, m))
  for (mi in 1:m) {
    idx <- sample(1:n, 1)
    xi <- data$X[idx, ]
    yi <- data$Y[idx]
    lpred <- sum(xi * theta.old)
    y.pred <- glm.model$h(lpred)  # link function of GLM
    w[mi+1] <- w[mi] - ai * (score(w[mi]) - score(theta.old) + mu)
  }
  # Assign SGD iterate to the last updated weight ("option I").
  theta.new <- w[m+1]
  return(theta.new)
}
ls.update <- function(i, data, theta.sgd) {
  # TODO: Generalize beyond Normal(0, A) data.
  n <- nrow(data$X)
  idx <- ifelse(i %% n == 0, n, i %% n) # sample index of data
  xi <- data$X[idx, ]
  theta.old <- theta.sgd[, i]
  return(data$obs.data$A %*% (xi - theta.old))
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
