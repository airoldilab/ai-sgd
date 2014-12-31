# This is a collection of auxiliary functions for usage in other scripts.

################################################################################
# Generalized linear models
################################################################################

logistic <- function(x) {
  # Return logit inverse.
  return(1/(1+exp(-x)))
}

logit <- function(x) {
  # Return logit.
  return(log(x/(1-x)))
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
# Observation matrix generation
################################################################################

generate.X.A <- function(n, p, lambdas=seq(0.01, 1, length.out=p)) {
  # Generate observations from Normal(0, A).
  #
  # Args:
  #   n: number of observations
  #   p: number of parameters
  #   lambdas: eigenvalues of A
  #
  # Returns:
  #   A list, where X is the n x p matrix and A is the covariance.
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
                          snr=1) {
  # Generate the dataset.
  #
  # Args:
  #   X.list: list whose element X is the design matrix, and whose other
  #           elements are any stored data used to generate X
  #   theta: true parameters
  #   glm.model: GLM model (see get.glm.model(..))
  #   snr: signal-to-noise ratio
  #     NOTE(ptoulis): cancelled SNR for now.
  #
  # Returns:
  #   The DATA object, which is a list with the following elements:
  #     Y = outcomes (n x 1)
  #     X = covariates (n x p)
  #     theta = true params. (p x 1)
  #     L = X * theta
  #     model = GLM model (see get.glm.model(..))
  #     obs.data = list containing any data used to generate X
  X <- X.list$X
  n <- nrow(X)
  p <- ncol(X)
  lpred <- X %*% theta
  # Generate outcomes according to the specified GLM.
  if (glm.model$name == "gaussian") {
    epsilon <- rnorm(n, mean=0, sd=1)
    # k <- sqrt(var(lpred)/(snr*var(epsilon)))
    y <- lpred + epsilon
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
  # TODO: Generalize this function beyond Normal(0, A) data.
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

run <- function(model, pars, n=1e4, p=1e1, add.methods=NULL, plot.save=F) {
  # Run AI-SGD for a set of parameters and any additionally selected methods,
  # and plot error over training data size. The set of parameters affect only
  # AI-SGD's learning rate.
  # TODO: Generalize this function beyond Normal(0, A) data.
  #
  # Args:
  #   model: the specified GLM
  #   pars: A npars x 2 matrix, where each row is a set of parameters to run
  #         AI-SGD on
  #   n: number of observations
  #   p: number of parameters
  #   add.methods: vector of additional methods to benchmark. Options are
  #                documented in sgd()
  #   plot.save: boolean specifying whether to save plot to disk or output it
  #
  # Returns:
  #   A ggplot object, plotting error over training data size for each
  #   optimization routine.
  set.seed(42)
  X.list <- generate.X.A(n, p)
  d <- generate.data(X.list,
                     glm.model=get.glm.model(model),
                     theta=2 * exp(-seq(1, p)))

  # Construct functions for learning rate.
  lr <- function(n, par) {
    # Ruppert's learning rate.
    # Note:
    # alpha / (alpha + n) = 1 / (1 + lambda0*n), where lambda0 = 1/alpha
    D <- par[1]
    alpha <- par[2]
    D*n^alpha
  }

  # Optimize!
  theta <- list()
  # Run AI-SGD for each set of parameters.
  pars.len <- ifelse(is.null(nrow(pars)), 1, nrow(pars)) # if pars is a vector
  for (i in 1:pars.len) {
    print("Running AI-SGD..")
    # Use parameters from the ith row, or pars itself if pars is a single set of
    # parameters.
    if (pars.len == 1) {
      par <- pars
    } else {
      par <- pars[i, ]
    }
    theta[[i]] <- sgd(d, sgd.method="AI-SGD", lr=lr, par=par)
  }
  names(theta) <- sprintf("AI-SGD, par #%i", 1:pars.len)
  # Run additionally specified methods.
  lr.explicit <- function(n, p) {
    gamma0 <- 1 / (sum(seq(0.01, 1, length.out=p)))
    alpha <- 1/0.01 # 1/minimal eigenvalue of Fisher information
    alpha/(alpha/gamma0 + n)
  }
  lr.implicit <- function(n) {
    alpha <- 1/0.01 # 1/minimal eigenvalue of Fisher information
    alpha/(alpha + n)
  }
  for (i in add.methods) {
    if (i %in% c("SGD", "ASGD", "LS-SGD")) {
      print(sprintf("Running %s..", i))
      theta[[i]] <- sgd(d, sgd.method=i, lr=lr.explicit)
    } else if (i %in% c("ISGD", "AI-SGD", "LS-ISGD")) {
      print(sprintf("Running %s..", i))
      theta[[i]] <- sgd(d, sgd.method=i, lr=lr.implicit)
    } else if (i == "Batch") {
      print(sprintf("Running %s..", i))
      theta[[i]] <- batch(d, sequence=round(10^seq(
        log(p + 10, base=10),
        log(n, base=10), length.out=100))
        ) # the sequence is equally spaced points on the log scale
    }
  }

  if (plot.save == TRUE) {
    # Plot and save image.
    png(sprintf("img/exp_%s_n%ip%i.png", model, log(n, base=10), log(p,
      base=10)), width=1280, height=720)
    plot.risk(d, theta)
    dev.off()
  } else {
    return(plot.risk(d, theta))
  }
}

benchmark <- function(n, p, rho,
                      methods=c("glmnet (naive)", "glmnet (cov)",
                                "ISGD", "AI-SGD"),
                      nreps=3) {
  # Benchmark stochastic gradient methods along with glmnet.
  # TODO: Generalize this function beyond correlated Normal data.
  #
  # Args:
  #   n: number of observations
  #   p: number of parameters
  #   rho: correlation
  #   methods: vector of methods to benchmark. Options are "glmnet (naive)",
  #            "glmnet (cov)", and all methods documented in sgd()
  #   nreps: number of replications
  #
  # Returns:
  #   A length(methods) x 4 data frame, with the following columns:
  #     name of method, average elapsed time, average mean squared error, number
  #     of replications
  library(glmnet)
  for (fn in c("sgd", "generate.X.corr", "generate.data")) {
    if(!exists(fn)) stop(sprintf("%s does not exist.", fn))
  }
  pb <- txtProgressBar(style=3)

  # Initialize results data frame (nmethods x 4).
  # Will return this object.
  nmethods <- length(methods)
  results <- as.data.frame(matrix(NA, nrow=nmethods, ncol=4))
  names(results) <- c("method", "time", "mse", "replications")
  results$method <- methods
  results$replications <- nreps

  # Initialize temporary matrices to store times and mean squared errors per
  # replication (nmethods x nreps).
  times <- matrix(NA, nrow=nmethods, ncol=nreps)
  mses <- matrix(NA, nrow=nmethods, ncol=nreps)

  # Initialize seeds.
  set.seed(42)
  seeds <- sample(1:1e9, size=nreps)

  # Construct auxiliary functions used for the methods.
  dist <- function(x, y) {
    # Calculate mean squared error.
    if (length(x) != length(y)) {
      stop("MSE should compare vectors of same length")
    }
    sqrt(mean((x-y)^2))
  }
  lr.explicit <- function(n, p, rho) {
    b <- rho/(1-rho)
    gamma0 <- 1/((b^2+1)*p)
    lambda0 <- 1
    alpha <- 1/lambda0
    return(alpha/(alpha/gamma0 + n))
  }
  lr.implicit <- function(n) {
    lambda0 <- 1
    alpha <- 1/lambda0
    return(alpha/(alpha + n))
  }

  # Run each simulation.
  for (i in 1:nreps) {
    # Set seed.
    set.seed(seeds[i])

    # Generate data.
    X.list <- generate.X.corr(n, p, rho=rho)
    theta <- ((-1)^(1:p))*exp(-2*((1:p)-1)/20)
    data <- generate.data(X.list, theta, snr=3)

    # Optimize!
    for (j in 1:nmethods) {
      if (methods[j] == "glmnet (naive)") {
        time.j <- system.time(
          {fit=glmnet(data$X, data$Y, alpha=1, standardize=FALSE,
           type.gaussian="naive")}
        )[1]
        mse.j <- median(apply(fit$beta, 2, function(est) dist(est, theta)))
      } else if (methods[j] == "glmnet (cov)") {
        time.j <- system.time(
          {fit=glmnet(data$X, data$Y, alpha=1, standardize=FALSE,
           type.gaussian="covariance")}
        )[1]
        mse.j <- median(apply(fit$beta, 2, function(est) dist(est, theta)))
      } else if (methods[j] %in% c("SGD", "ASGD", "LS-SGD")) {
        time.j <- system.time(
          {fit=sgd(data, sgd.method=methods[j], lr=lr.explicit, rho=rho)}
        )[1]
        mse.j <- dist(fit[, ncol(fit)], theta)
      } else if (methods[j] %in% c("ISGD", "AI-SGD", "LS-ISGD")) {
        time.j <- system.time(
          {fit=sgd(data, sgd.method=methods[j], lr=lr.implicit)}
        )[1]
        mse.j <- dist(fit[, ncol(fit)], theta)
      }
      times[j, i] <- time.j
      mses[j, i] <- mse.j
      setTxtProgressBar(pb, (i*nmethods+j)/(nreps*nmethods))
    }
  }
  # Take means of the simulation reults.
  results$time <- rowMeans(times)
  results$mse <- rowMeans(mses)
  print("") # print newline
  return(results)
}

################################################################################
# Miscellaneous
################################################################################

fracSec <- function() {
  # Generate a seed number based on the current time.
  now <- as.vector(as.POSIXct(Sys.time())) / 1000
  as.integer(abs(now - trunc(now)) * 10^8)
}
interval.map <- function(a, b, c, d, x) {
  # Scale values in [a,b] to [c,d].
  return(c + (d-c)/(b-a) * (x-a))
}
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
