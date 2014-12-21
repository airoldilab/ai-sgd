## Notes from the Polyak paper.
p = 2
# Define some matrices.
A.eig = seq(0.05, 5, length.out=p)
lmin = 2 / min(A.eig)
Lmax = 2 / max(A.eig)
U = qr.Q(qr(matrix(runif(p^2), nrow=p)))
A = U %*% diag(A.eig) %*% t(U)
I = diag(p)

gamma.j <- function(j, gamma=2/lmin, alpha=0.8) {
  # return(gamma)
  1 * (1+j)^(-0.05)
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

phi.jt <- function(j, t) {
  solve(A) - Xjt.bar(j,t)
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
      plot(svector, type="l", main=sprintf("t = %d, Current min=%.2f", t, min(svector)))
    }
  }
  print(tail(svector))
}
