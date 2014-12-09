# An implementation of stochastic gradient methods for GLMs.

sgd <- function(data, sgd.method, averaged=F, ls=F, lr, ...) {
  # Find the optimal parameter values using a stochastic gradient method for a
  # generalized linear model.
  #
  # Args:
  #   data: DATA object created through sample.data(..) (see functions.R)
  #   sgd.method: "explicit" or "implicit"
  #   averaged: boolean specifying whether to average estimates
  #   ls: boolean specifying whether to use least squares estimate
  #   lr: function which computes learning rate with input the iterate index
  #
  # Returns:
  #   A p x n matrix where the jth column is the jth theta update.

  # Check input.
  stopifnot(
    all(is.element(c("X", "Y", "model"), names(data))),
    sgd.method %in% c("explicit", "implicit")
  )
  n <- nrow(data$X)
  p <- ncol(data$X)
  glm.model <- data$model
  # Initialize parameter matrix for sgd (p x n).
  # Will return this matrix.
  theta.sgd <- matrix(0, nrow=p, ncol=n)
  # Initialize y matrix if the least squares estimate is desired (p x n).
  y <- NULL
  ai <- NULL
  theta.new <- NULL
  if (ls == TRUE) {
    y <- matrix(0, nrow=p, ncol=n+1)
  }
  # Main iteration: i = #sample index.
  # Assumes: y, ai,   Updates: theta.new
  for (i in 2:n) {
    xi <- data$X[i, ]
    yi <- data$Y[i]
    theta.old <- theta.sgd[, i-1]

    # Make computation easier.
    xi.norm <- sum(xi^2)
    lpred <- sum(xi * theta.old)
    y.pred <- glm.model$h(lpred)  # link function of GLM

    # Calculate learning rate.
    if (sgd.method == "explicit") {
      ai <- lr(i, p, ...)
    } else if (sgd.method == "implicit") {
      ai <- lr(i, ...)
    }

    # Make the update.
    if (ls == TRUE) {
      y[, i] <- data$A %*% (xi - theta.old)
    }

    if (sgd.method == "explicit") {
      theta.new <- theta.old + ai * (yi - y.pred) * xi
    } else if (sgd.method == "implicit") {
      # 1. Define the search interval.
      ri <- ai * (yi - y.pred)
      Bi <- c(0, ri)
      if(ri < 0) {
        Bi <- c(ri, 0)
      }

      implicit.fn <- function(u) {
        u - ai * (yi - glm.model$h(lpred + xi.norm * u))
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
    }

    theta.sgd[, i] <- theta.new
  }

  if (averaged == TRUE) {
    # Average over all estimates.
    theta.sgd <- t(apply(theta.sgd, 1, function(x) {
      cumsum(x)/(1:length(x))
    }))
  }

  if (ls == TRUE) {
    # Run least squares fit over all estimates.
    beta.0 <- matrix(0, nrow=p, ncol=n)
    beta.1 <- matrix(0, nrow=p, ncol=n)
    for (i in 2:n) {
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
