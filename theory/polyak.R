## Notes from the Polyak paper.
p = 3
# Define some matrices.
lambdaA.min <- 0.2
lambdaA.max <- 3
A.eig = seq(lambdaA.min, lambdaA.max, length.out=p)
U = qr.Q(qr(matrix(runif(p^2), nrow=p)))
A = U %*% diag(A.eig) %*% t(U)
A.inv = solve(A)
I = diag(p)

solve.lyapunov <- function() {
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

gamma.j <- function(j, gamma=2/lambdaA.min, alpha=0.8) {
  # return(gamma)
  1 * (1+j)^(-0.4)
}

Gamma.sum <- function(j, t) {
  sum(gamma.j(j:t))
}

Xjt <- function(j, t) {
  # Computes X_j^t defined in (A1)
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


Xjt.bar <- function(j, t) {
  stopifnot(t >= j)
  return(gamma.j(j) * sum.Xjt(j, t))
}

average.Xjt.bar <- function(tmax) {
  S = 0
  conv <- c()
  for(j in 1:tmax) {
    S <- S + Xjt.bar(j, tmax)
    conv <- c(conv, norm(S/j - A.inv, type="F"))
    if(j %% 50 == 0)
      plot(log(conv), type="l", main="Convergence of log(||Xjt.bar - A^-1||)")
  }
  print(tail(conv))
  return(S/tmax)
}

phi.jt <- function(j, t) {
  solve(A) - Xjt.bar(j,t)
}

check.Xjt.frobenius <- function(j=5, tmax=500) {
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
    X = Xjt(j, t)
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

check.Xjt.convergence <- function(tmax) {
  S = matrix(0, nrow=p, ncol=p)
  svector = c()
  for(t in 1:tmax) {
    S = matrix(0, nrow=p, ncol=p)
    for(j in 0:t) {
      S <- S + phi.jt(j, t)
    }
    S.avg = S/t
    svector = c(svector, mean(S.avg^2))
    if(t %% 20 == 0) {
      plot(svector, type="l", main=sprintf("t = %d, Current min=%.4f", t, min(svector)))
    }
  }
  print(tail(svector))
}
