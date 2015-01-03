# An implementation of stochastic gradient methods for GLMs.

sgd <- function(data, sgd.method, lr, npass=1, lambda=0, ...) {
  # Find the optimal parameters using a stochastic gradient method for
  # generalized linear models.
  #
  # Args:
  #   data: DATA object created through sample.data(..) (see functions.R)
  #   sgd.method: a string which is one of the following: "SGD", "ASGD",
  #               "LS-SGD", "ISGD", "AI-SGD", "LS-ISGD"
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
    sgd.method %in% c("SGD", "ASGD", "LS-SGD", "ISGD", "AI-SGD", "LS-ISGD")
  )
  n <- nrow(data$X)
  p <- ncol(data$X)
  glm.model <- data$model
  # Initialize parameter matrix for the stochastic gradient descent (p x n*npass+1).
  # Will return this matrix.
  theta.sgd <- matrix(0, nrow=p, ncol=n*npass+1)
  theta.new <- NULL
  ai <- NULL
  # Initialize y matrix if method uses least squares estimate (p x n*npass+1).
  y <- NULL
  if (sgd.method %in% c("LS-SGD", "LS-ISGD")) {
    y <- matrix(0, nrow=p, ncol=n*npass+1)
  }
  # Main iteration: i = #iteration
  # Assumes: y, ai,   Updates: theta.new
  for (i in 1:(n*npass)) {
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

    # Make the update.
    if (sgd.method %in% c("LS-SGD", "LS-ISGD")) {
      y[, i] <- data$A %*% (xi - theta.old)
    }

    if (sgd.method %in% c("SGD", "ASGD", "LS-SGD")) {
      # TODO: Test to see if the regularization actually works.
      #theta.new <- theta.old + ai * (yi - y.pred) * xi
      theta.new <- theta.old + ai * ((yi - y.pred) * xi + lambda*norm(theta.old, type="2"))
    } else if (sgd.method %in% c("ISGD", "AI-SGD", "LS-ISGD")) {
      theta.new <- rep(NA, p)
      # 1. Define the search interval.
      #ri <- ai * (yi - y.pred)
      ri <- ai * ((yi - y.pred) + lambda*norm(theta.old, type="2"))
      Bi <- c(0, ri)
      if (ri < 0) {
        Bi <- c(ri, 0)
      }

      implicit.fn <- function(u) {
        #u - ai * (yi - glm.model$h(lpred + xi.norm * u))
        u - ai * ((yi - glm.model$h(lpred + xi.norm * u)) + lambda*norm(u, type="2"))
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
      #theta.new <- theta.old + ksi.new
    }

    theta.sgd[, i+1] <- theta.new
  }

  if (sgd.method %in% c("ASGD", "AI-SGD")) {
    # Average over all estimates.
    theta.sgd <- t(apply(theta.sgd, 1, function(x) {
      cumsum(x)/(1:length(x))
    }))
  }
  if (sgd.method %in% c("LS-SGD", "LS-ISGD")) {
    # Run least squares fit over all estimates.
    beta.0 <- matrix(0, nrow=p, ncol=n*npass+1)
    beta.1 <- matrix(0, nrow=p, ncol=n*npass+1)
    for (i in 1:(n*npass+1)) {
      x.i <- theta.sgd[, 1:i]
      y.i <- y[, 1:i]
      bar.x.i <- rowMeans(x.i)
      bar.y.i <- rowMeans(y.i)
      beta.1[, i] <- rowSums(y.i*(x.i - bar.x.i))/rowSums((x.i - bar.x.i)^2)
      beta.0[, i] <- bar.y.i - beta.1[, i] * bar.x.i
    }
    theta.sgd <- -beta.0/beta.1
  }

  return(theta.sgd)
}
