# An implementation of batch gradient descent for GLMs.

batch <- function(data, sequence=1:nrow(data$X), intercept=F, slope=T) {
  # Find the optimal parameter values using batch gradient descent for a
  # generalized linear model.
  #
  # Args:
  #   data: DATA object created through sample.data(..) (see functions.R)
  #   sequence: Vector of iterate indices to calculate batch for. This is in
  #             case you only aim to run batch on a subset of indices.
  #             Default is all indices.
  #   intercept: boolean specifying whether to include intercept term for regression
  #   slope: boolean specifying whether to include slope term for regression
  #
  # Returns:
  #   A p x length(sequence) matrix where the jth column is the theta update
  #   with iterate index sequence[j].

  # Check input.
  stopifnot(
    all(is.element(c("X", "Y", "model"), names(data))),
    length(sequence) > 0, sequence[1] > 1
  )
  p <- ncol(data$X)
  glm.model <- data$model
  # Return the mean if one desires to fit with a zero slope.
  # NOTE(ptoulis): This was a bit confusing. Where are we using no-slope?
  #if (slope == FALSE) {
  # theta.batch <- t(apply(data$X, 2, function(x) {
  #   cumsum(x)/(1:length(x))
  #   }))
  # theta.batch <- theta.batch[, sequence]
  # colnames(theta.batch) <- sequence
  # return(theta.batch)
  #}
  # Initialize parameter matrix for batch (p x length(sequence)).
  # Will return this matrix.
  theta.batch <- matrix(0, nrow=p, ncol=length(sequence))
  colnames(theta.batch) <- sequence
  # Include or exclude the intercept term in the regression formula.
  if (intercept == FALSE) {
    glm.formula <- "y ~ X+0"
  } else {
    glm.formula <- "y ~ X"
  }
  # Main iteration: idx = index to store the ith update
  for (idx in 1:length(sequence)) {
    i <- sequence[idx]
    X <- data$X[1:i, ]
    y <- data$Y[1:i]

    # Make the update.
    if (glm.model$name == "gaussian") {
      theta.new <- glm(as.formula(glm.formula), family=gaussian)$coefficients
    } else if (glm.model$name == "poisson") {
      theta.new <- glm(as.formula(glm.formula), family=poisson)$coefficients
    } else if (glm.model$name == "logistic") {
      theta.new <- glm(as.formula(glm.formula), family=binomial)$coefficients
    }

    theta.batch[, idx] <- theta.new
  }

  return(theta.batch)
}
