## Notes from the Polyak paper.
rm(list=ls())
## Problem definition: Linear case.
p = 3
lambdaA.min <- 0.1
lambdaA.max <- 3
gamma.rate <- 0.4
A.eig = seq(lambdaA.min, lambdaA.max, length.out=p)
U = qr.Q(qr(matrix(runif(p^2), nrow=p)))
A = U %*% diag(A.eig) %*% t(U)
A.inv = solve(A)
I = diag(p)
M0 = matrix(0, nrow=p, ncol=p)

solve.lyapunov <- function() {
  # Solve A^T V + V A = I.
  V = I
  converge = F
  B = solve(I + A)
  At = t(A)
  while(!converge) {
    V.new = B %*% (I + V - At %*% V)
    converge = norm(V.new - V, type="F") < 1e-5
    V = V.new
  }
  return(V)
}

gamma.j <- function(j, exponent=0.5, gamma_0=1) {
  # Learning rate
  gamma_0 * j^(-exponent)
}

Gamma.sum <- function(j, t) {
  # Compute the sum of rates.
  sum(gamma.j(j:t))
}

bound.implicit.gamma <- function(exponent, N=1e6, lam=2.5) {
  g = gamma.j(1:N, exponent=exponent)
  x = 1 + lam * g
  K = -1e-4 + (1/lam) * min(log(x) / g)
  bound = exp(-K * lam * g)
  print("Bound - actual")
  print(head(bound), 10)
  print(head(1/x), 10)
  print((1/lam) * head(log(x)/g))
  print("Learning rates...")
  print(head(g))
  print(sprintf("Lambda = %.3f, K = %.3f, 1/L=%.3f Limit min=%.3f", 
                lam, K, 1/lam, tail(log(x) / g, 1)))
  plot(1/x, type="l", col="green")
  lines(bound, col="red")
  print(sprintf("Total correct bounds = %.3f%%", 100 * sum(bound > 1/x) / N))
  
  ## Now check the bound.
  prods = cumprod(1/x)
  bounds = exp(-K * lam * cumsum(g))
  plot(prods, bounds, type="l", lty=3)
  lines(prods, prods, col="red")
  print("Tail of partial products - bounds")
  print(tail(prods))
  print(tail(bounds))
}

Dot.term <- function(exponent, N=1e5, lam=2) {
  # learning rates.
  g = c(1, gamma.j(1:N, exponent=exponent))
  f = 1 / (1 + lam * g)
  fprod = cumprod(f)
  tvals = seq(10, N, length.out=1000)
  
  D0t = sapply(tvals, function(t) sum(head(fprod, t)))
  plot(tvals, D0t, type="l")
  print(tail(D0t))
}

implicit.theory.normal <- function(gamma=2) {
  # Code to check some of the theoretical claims.
  run.im <- function(data.y, plot=F) {
    mu.im = 0
    alpha = 1 / (1+gamma)
    for(i in 1:length(data.y)) {
      mu.im = alpha * mu.im + (1-alpha) * data.y[i]
    }
    return(mu.im)
  }
  
  nreps = 500
  mu = 4.5
  sigma = 1.4
  nsamples = 1e4
  theta.im = c()
  for(j in 1:nreps) {
    y = rnorm(nsamples, mean=mu, sd=sigma)
    theta.im[j] <- run.im(y)
  }
  plot(hist(theta.im))
  abline(v=mu, col="red")
  theoretical.var = sigma^2 * gamma / (2 + gamma)
  print(sprintf("Variance of implicit estimates = %.3f / Theoretical = %.3f", 
                var(theta.im), theoretical.var))
}

Xjt <- function(j, t) {
  # Computes X_j^t defined in (A1)
  # Xj(t+1) = (I - g_t A) Xjt
  # thus, Xj(t+1) = (I-gt A) * (I-g(t-1) A) * ... (I - g(j) A)
  stopifnot(t >= j)
  if(t==j) return(I)
  return( (I - gamma.j(t-1) * A) %*% Xjt(j, t-1))
}
sum.Xjt <- function(j, upto) {
  # Computes sum_{i=j}^{i=upto} Xj^i
  S = matrix(0, nrow=p, ncol=p)
  X = I
  if(upto < j) return(S)
  for(i in j:upto) {
    S <- S + X
    X <- (I - gamma.j(i) * A) %*% X
  }
  return(S)
}

Yjt <- function(j, t) {
  # Computes the implicit counterpart of Xjt.
  # Yj(t+1) = (I + g_t A)^-1 Yjt
  stopifnot(t >= j)
  if(t==j) return(I)
  return(solve(I + gamma.j(t-1) * A) %*% Yjt(j, t-1))
}

sum.Yjt <- function(j, upto) {
  # Computes sum_{i=j}^{i=upto} Yj^i
  S = M0
  Y = I
  if(upto < j) return(S)
  for(i in j:upto) {
    S <- S + Y
    Y <- solve(I + gamma.j(i) * A) %*% Y
  }
  return(S)
}

## Averaged versions.
Xjt.bar <- function(j, t) {
  if(t < j + 1) return(matrix(0, nrow=p, ncol=p))
  return(gamma.j(j) * sum.Xjt(j, t-1))
}
Yjt.bar <- function(j, t) {
  if(t < j + 1) return(matrix(0, nrow=p, ncol=p))
  return(gamma.j(j) * sum.Yjt(j, t-1))
}

###    Check theoretical claims.
## Check that sum_j bar{Xjt} - A^-1 = o(t)
# i.e., that the Xjt.bar, Yjt.bar are getting close to A^-1
check.bar.convergence <- function(tmax, is.X=T, by=50, plot.last=100) {
  S = M0
  j = 50
  norms.vals <- c()
  plot.points = seq(j, tmax, by=40)
  for(t in j:tmax) {
    if(is.X) {
      S = (A.inv - Xjt.bar(j, t))
    } else {
      S = (A.inv - Yjt.bar(j, t))
    }
    norms.vals <- c(norms.vals, norm(S/t, type="F"))
    if(t %in% plot.points) {
      plot(norms.vals, type="l", main=sprintf("t=%d. Check claim on averaged %sjt. Last=%.4f", 
                                                         t, ifelse(is.X, "X", "Y"), tail(norms.vals, 1)))
    }
  }
  print(tail(norms.vals))
}

check.frobenius.norm <- function(j=5, tmax=500) {
  tvalues = seq(10, tmax, by=20)
  real.norm <- c()
  upper.bound <- c()
  polyak.bound <- c()
  V = solve.lyapunov()
  lambda.min = min(eigen(A)$values)
  lamV.min = min(eigen(V)$values)
  lamV.max = max(eigen(V)$values)
  lambdaA = eigen(A)$values
  for(t in tvalues) {
    X = Yjt(j, t)
    real.norm <- c(real.norm, norm(X, type="F"))
    upper.bound <- c(upper.bound, sqrt(sum(exp(-2 *lambdaA * Gamma.sum(j, t-1)))))
    polyak.bound <- c(polyak.bound, sqrt(lamV.max / lamV.min) * exp((-0.5/lamV.max) * Gamma.sum(j, t-1)))
  }
  stopifnot(all(diff(real.norm) < 0))
  plot(real.norm, upper.bound, type="l", col="red", xlim=c(0, max(real.norm)))
  lines(real.norm, polyak.bound, col="green")
  lines(real.norm, real.norm, lty=3)

  print(round(real.norm, 3))
  print(round(upper.bound, 3))
  print(all(upper.bound >= real.norm))
}

partial.products <- function(t, alpha, beta) {
  # compute 
  # 1 / (1 + a * 1^-b) * (1+ a 2^-b)...(1+a t^-b)
  # Is this equal to exp(-t + G(t))  where $
  gamma = seq(1, t)^(-beta)
  frac1 = 1 / (1 + gamma * alpha)
  frac2 = gamma * alpha * frac1
 
  P = cumprod(frac1)
  G = cumsum(frac2)
  
  
  plot(log(P), type="l")
  lines(-G, type="l", col="red")

}
