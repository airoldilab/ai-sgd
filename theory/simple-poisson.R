# Example Poisson
rm(list=ls())
true.theta = c(2.2, 0.4)

sample.data <- function(n) {
  theta.true = true.theta
  X = matrix(NA, nrow=n, ncol=2)
  y = rep(NA, n)
  for(i in 1:n) {
    u = sample(3, size=1, prob=c(0.6, 0.2, 0.2))
    x = (u==1) * c(0, 1) + (u==2) * c(1, 0) + (u==3) * c(1, 1)
    X[i, ] <- as.numeric(x)
    y[i] <- rpois(1, lambda=exp(sum(X[i, ] * theta.true)))
  }
  return(list(X=X, y=y, theta.true=theta.true))
}

optimal.alpha <- function() {
  Fisher = 0.2 * diag(exp(true.theta))
  lam = eigen(Fisher)$values
  
  f <- function(a) sum(a^2 / (2 * a * lam - 1))
  amin = 1/ (2*min(lam)) + 1e-3
  alpha.opt = optim(par=c(amin), fn=f, method="L-BFGS-B", lower=amin, upper=Inf)$value
  print(alpha.opt)
}

sgd <- function(data, lr=function(t) 1/(1+t), verbose=F) {
  # Runs explicit SG.D
  n = nrow(data$X)
  p = ncol(data$X)
  theta.old = matrix(20, nrow=p,  ncol=1)
  theta.all = matrix(NA, nrow=p, ncol=0)
  risk = c()
  X = data$X
  y = data$y
  
  plot.until <- function(arg_t, a, b, c) {
    par(mfrow=c(2, 2))
    plot(a[1:arg_t], main="Total risk", type="l")

    plot(b[1, 1:arg_t], type="l")
    abline(h=c[1], col="red")
    plot(b[2, 1:arg_t], type="l")
    abline(h=c[2], col="red")
    
  }
  
  plot.points = as.integer(seq(1, n, length.out=0.1 * n))
  if(!verbose)  {
    plot.points=c()
  }
  for(i in 1:n) {
    yi = y[i]
    xi = matrix(X[i, ], ncol=1)
    yi.pred = as.numeric(exp(t(theta.old) %*% xi))
    theta.old = theta.old + lr(i) * (yi - yi.pred) * xi
    if(verbose) {
      theta.all <- cbind(theta.all, theta.old)
      risk = c(risk, norm(theta.old-data$theta.true, type="F"))
      if(i %in% plot.points) {
        plot.until(i, risk, theta.all, data$theta.true)
      }
    }
  }
  
  return(theta.old)
}

mse <- function(est) {
  x = est - matrix(true.theta, nrow=nrow(est), ncol=ncol(est), byrow=T)
  # print(head(x))
  t(x) %*% x / nrow(x)
}

sgd.mse <- function(nreps, method="explicit", n=1e4, alpha=1) {
  # store the SGD iterates.
  theta = matrix(NA, nrow=nreps, ncol=2)
  # Optimal theta
  theta.opt = matrix(NA, nrow=nreps, ncol=2)
  pb = txtProgressBar(style=3)
  for(j in 1:nreps) {
    d = sample.data(n)
    
    if(method=="explicit") {
      theta[j, ] <- sgd(d, alpha)
    } else {
      theta[j, ] <- sgd.implicit(d, alpha)
    }
    
    theta.opt[j, ] <- glm(d$y~d$X + 0, family="poisson")$coefficients
    setTxtProgressBar(pb, value=j/nreps)
  }
  # Theoretical variance
  d = sample.data(10)
  Fisher = 0.2 * diag(exp(d$theta.true))
  theoretical = alpha^2 * solve(2 * alpha * Fisher - diag(2)) %*% Fisher
  print(sprintf("theoretical MSE method %s", method))
  print(theoretical)
  print(sprintf("Empirical for method %s ", method))
  print(mse(theta) * n)
  ## Optimal
  print("Optimal ")
  print(solve(Fisher))
  print("Observed optimal")
  print(n * mse(theta.opt))
  
}

# Stuff for the implicit method.
implicit.accelerated.ai <- function(yi, xi, ai, theta.old, gamma=0.5) {
  ksi = implicit.update(yi, xi, ai, theta.old, return.ksi=T)
  bi.left = 0
  bi.right = ai
  bi = (bi.left + bi.right)/2
  ksi.b = implicit.update(yi, xi, bi, theta.old, return.ksi=T)
  G = gamma * ksi
  while(abs(ksi.b - G) > 1e-5) {
    if(ksi.b > G) {
      if(ksi > 0)
        bi.right = bi
      else
        bi.left = bi
    } else {
      if(ksi > 0)
        bi.left = bi
      else
        bi.right = bi
    }
    bi = (bi.left + bi.right)/2
    ksi.b = implicit.update(yi, xi, bi, theta.old, return.ksi=T)
    if(abs(bi.left - bi.right) < 1e-8) {
      warning("Numerical instability")
      return(bi.left)
    }
  }
  
  return(bi)
}

test.implicit.accelerated.ai <- function() {
  yi = 3
  xi = c(0, 1)
  theta.old = c(20, 20)
  ai = 0.1

  Delta.i = implicit.update(yi, xi, ai, theta.old) - theta.old
  
  bi = implicit.accelerated.ai(yi, xi, ai, theta.old, gamma = 1-1e-5)
  Delta.j = implicit.update(yi, xi, bi, theta.old) - theta.old
  
  print("Initial Delta")
  print(Delta.i)
  print("1/gamma-Delta.")
  print(Delta.j)
  print("Final rate")
  print(bi)
}

implicit.update <- function(yi, xi, ai, theta.old, return.ksi=F) {
  # Computes the implicit update.
  #
  # Args:
  #  yi, xi, ai = observation, covariate vector, learning rate, resp.
  #  theta.old = vector of previous iterate.
  #
  norm.xi = sum(xi^2)
  get.score.coeff <- function(ksi) {
    return(yi - exp(sum(theta.old * xi) + norm.xi * ksi))
  }
  # 1. Define the search interval
  ri = ai * get.score.coeff(0)
  Bi = c(0, ri)
  if(ri < 0) {
    Bi <- c(ri, 0)
  }
  
  implicit.fn <- function(u) {
    u  - ai * get.score.coeff(u)
  }
  # 2. Solve implicit equation
  xit = NA
  if(Bi[2] != Bi[1])
    xit = uniroot(implicit.fn, interval=Bi)$root
  else 
    xit = Bi[1]
  
  if(return.ksi) {
    return(xit)
  }
  
  theta.old + xit * xi
}

sgd.implicit <- function(data, lr=function(t)   1 / (1+t),
                         verbose=F,
                         averaged=F, accelerated=F) {
  n = nrow(data$X)
  p = ncol(data$X)
  theta.old = matrix(20, nrow=p,  ncol=1)
  theta.all = matrix(NA, nrow=p, ncol=0)
  theta.sum = matrix(0, nrow=p, ncol=1)
  risk = c()
  
  X = data$X
  y = data$y
  
  # Plot stuff.
  plot.until <- function(arg_t, a, b, c) {
    par(mfrow=c(2, 2))
    plot(a[1:arg_t], main="Total risk", type="l")
    
    plot(b[1, 1:arg_t], type="l")
    abline(h=c[1], col="red")
    plot(b[2, 1:arg_t], type="l")
    abline(h=c[2], col="red")
    
  }  
  plot.points = as.integer(seq(1, n, length.out=0.05 * n))
  
  # Main loop: iterate over all points.
  acc.gamma = 1-1e-3
  
  for(i in 1:n) {
    yi = y[i]
    xi = matrix(X[i, ], ncol=1)
    ai = lr(i)
    # Implicit update.
    theta.new = implicit.update(yi, xi, ai, theta.old)
    if(accelerated) {
      bi = implicit.accelerated.ai(yi, xi, ai, theta.old, gamma = acc.gamma)
      m = theta.new - theta.old
      f = - (ai * bi) * (1-acc.gamma)  / (bi - acc.gamma * ai)
      if(is.finite(f)) {
        theta.old = theta.old + (1/i) * f * m / ai
        # print(sprintf("f = %.3f", f))
        # print(theta.old)
        #print(sprintf("ai = %.3f bi=%.3f", ai, bi))
        
      } else {
        theta.old = theta.new
      }
      if(any(is.na(theta.old)))
        stop("Error with NAs")
    } else { 
      theta.old = theta.new
    }
    # Used for the averaged version
    theta.sum = theta.sum + theta.old
    ## plotting if any
    if(verbose) {
      theta.all <- cbind(theta.all, theta.old)
      risk = c(risk, norm(theta.old-data$theta.true, type="F"))
      if(i %in% plot.points) {
        plot.until(i, log(risk), theta.all, data$theta.true)
      }
    }
  }
  # Return the average if asked for.
  if(averaged) {
    return((1/n) * theta.sum)
  }
  return(theta.old)
}

sgd.implicit.var <- function() {
  
}

