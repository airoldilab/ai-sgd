# Stochastic gradient function to be used in both Friedman et. al, section 5.1,
# and Xu, section 6.2.

sgd <- function(data, method, averaged=F, ls=F, lr, ...) {
  # Find the optimal parameter values using a stochastic gradient method for a
  # linear model.
  #
  # Args:
  #   data: List of X, Y in particular form
  #   method: "explicit" or "implicit"
  #   averaged: boolean of whether or not to average estimates
  #   ls: boolean of whether or not to use least squares estimate
  #   lr: function which computes learning rate with input the iterate index
  #
  # Returns:
  #   p x (n+1) matrix where the jth column is the jth theta update

  # check.data(data)
  n <- nrow(data$X)
  p <- ncol(data$X)
  # Initialize parameter matrix for sgd (p x iters).
  theta.sgd <- matrix(0, nrow=p, ncol=n+1)
  # Initialize y matrix if least squares estimate desired (p x iters).
  if (ls == TRUE) y <- matrix(0, nrow=p, ncol=n+1)

  for (i in 1:n) {
    xi <- data$X[i, ]
    theta.old <- theta.sgd[, i]

    # Compute learning rate.
    if (method == "explicit") {
      ai <- lr(i, p, ...)
    } else if (method == "implicit") {
      ai <- lr(i, ...)
    }

    # Make computation easier.
    xi.norm <- sum(xi^2)
    lpred <- sum(theta.old * xi)
    fi <- 1 / (1 + ai * sum(xi^2))
    yi <- data$Y[i]

    # Update.
    if (ls == TRUE) y[, i+1] <- data$A %*% (xi - theta.old)
    if (method == "explicit") {
      theta.new <- (theta.old - ai * lpred * xi) + ai * yi * xi
    } else if (method == "implicit") {
      theta.new <- theta.old  - ai * fi * (lpred - yi) * xi
    }

    theta.sgd[, i+1] <- theta.new
  }

  if (averaged == TRUE) {
    # Average over all estimates.
    theta.sgd <- t(apply(theta.sgd, 1, function(x) {
      cumsum(x)/(1:length(x))
    }))
  }

  if (ls == TRUE) {
    # Run least squares fit over all estimates.
    beta.0 <- matrix(0, nrow=p, ncol=n+1)
    beta.1 <- matrix(0, nrow=p, ncol=n+1)
    for (i in 2:(n+1)) {
      x.i <- theta.sgd[, 1:i]
      y.i <- y[, 1:i]
      bar.x.i <- rowMeans(x.i)
      bar.y.i <- rowMeans(y.i)
      beta.1[, i] <- rowSums(y.i*(x.i - bar.x.i))/rowSums((x.i - bar.x.i)^2)
      beta.0[, i] <- bar.y.i - beta.1[, i] * bar.x.i
    }
    theta.sgd <- -beta.0/beta.1
    # TODO: Cleanup?
    # his the slower method but more readable and also slightly
    # more accurate in numerical precision(?). They disagree by 1e-4.
    #theta.sgd.ls <- matrix(0, nrow=p, ncol=n+1)
    #for (i in 2:(n+1)) {
      #for (j in 1:p) {
      #  y.i <- y[j, 1:i]
      #  x.i <- theta.sgd[j, 1:i]
      #  lm.est <- lm(y.i~x.i)$coefficients
      #  theta.sgd.ls[j, i] <- -lm.est[1]/lm.est[2]
      #}
    #}
    #theta.sgd <- theta.sgd.ls
  }

  return(theta.sgd)
}
