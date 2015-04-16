## Example: Cox PH model..
rm(list=ls())
library(survival)

generate.data <- function(n, p) {
  X = matrix(rbinom(n * p, size=2, prob = 0.1), nrow=n, ncol=p)
  # X[, 1] <- 1
  beta = exp(-(1/sqrt(p)) * seq(1, p))
  pred = apply(X, 1, function(r) exp(sum(r * beta)))
  Y = rexp(n, rate = pred)
  
  q3 = quantile(Y, prob=c(0.8))  # Q3 of Y
  epsilon = 0.001 # probability of censoring smallest Y
  k = log(1/epsilon - 1) / (q3 - min(Y))
  censor.prob = (1 + exp(-k * (Y-q3)))**(-1)
  # plot(censor.prob, main="Censoring probabilities", type="l")
  C = rbinom(n, size=1, prob= censor.prob)
  return(list(x=X, y=Y, censor=C, beta=beta))
}

dist <- function(x, y)  {
 sqrt(mean((x-y)**2))  
}

fit.cox <- function(data) {
  fit <- coxph(Surv(y, censor) ~ x, data)
  print(names(fit))
  print(summary(fit))
  print("real parameters")
  print(data$beta)
  print("Distance")
  print(dist(fit$coeff, data$beta))
}

cox.sgd <- function(data, niters=1e3, C=1, implicit=F, averaging=F) {
  if(implicit) {
    print("Running Implicit SGD for Cox PH model.")
  }
  n = length(data$y)
  p = ncol(data$x)
  beta = matrix(0, nrow=p, ncol=1)
  gammas = C / seq(1, niters)
  if(averaging) {
    gammas = C / seq(1, niters)**(0.35)
  }
  print(summary(gammas))
  betas = matrix(0, nrow=p, ncol=0)
  mse = c()
  
  # order units based on event times.
  ord = order(data$y)
  d = data$censor[ord]  # censor
  x = matrix(data$x[ord, ], ncol=p)  # ordered covariates.
  
  # plotting.
  plot.points = as.integer(seq(1, niters, length.out=20))
  pb = txtProgressBar(style=3)
  
  # params for the implicit method.
  fj <- NA
  fim <- NA
  lam <- 1
  # param for averaging
  beta.bar <- matrix(0, nrow=p, ncol=1)
  
  units.sample = sample(1:n, size=niters, replace=T)
  for(i in 1:niters) {
    gamma_i = gammas[i]
    ksi = exp(x %*% beta)
    j = units.sample[i] # sample unit
    Xj = matrix(x[j, ], ncol=1) # get covariates
    Hj = sum(head(d, j) / head(rev(cumsum(rev(ksi))), j))  # baseline hazards for units in risk set Rj.
    
    # Defined for the implicit
    if(implicit) {
      fj <- function(b) {
        a = d[j] - Hj * exp(b)
        if(a==-Inf) return(-1e5)
        if(a==Inf) return(1e5)
        return(a)
      }    
      pred = sum(x[j, ] * beta)
      Xj.norm = sum(Xj**2)
      fim <- function(el) {
        #  print(el)
        a = el * fj(pred) - fj(pred + gamma_i * Xj.norm * el * fj(pred))
        if(a < -1e100) return(-1e5)
        if(a > 1e100) return(1e5)
        return(a)
      }
      
      lam = optim(par=c(0), f = function(b) fim(b)**2, method = "L-BFGS")$par
    }
     
    # Update. (lam=1 for explicit -- updated for implicit)
    beta = beta + gamma_i * (d[j] - Hj * ksi[j]) * Xj
    
    if(averaging) {
      beta.bar = (1/i) * ((i-1) * beta.bar + beta)
      mse <- c(mse, dist(beta.bar, data$beta))
    } else {
      mse <- c(mse, dist(beta, data$beta))
    }
    
    if(i %in% plot.points)
      plot(mse, type="l", main=sprintf("dist=%.4f (implicit=%d)", tail(mse, 1), implicit), ylim=c(0, max(mse)))
    
    setTxtProgressBar(pb, value=i/niters)
  }
  print("SGD params")
  print(beta)
  print("Distance")
  print(dist(beta, data$beta))
}


cox.implicit <- function(niters=1e4) {
  beta = 0
  betas.im = c(0)
  # candidates <- seq(-100, 100, length.out=100)
  lams = c()
  for(i in 1:niters) {
    ksi = exp(X * beta)
    gamma_i = gammas[i]
    j = sample(1:n, size=1)
    Hj = sum(head(d, j) / head(rev(cumsum(rev(ksi))), j))
    fj <- function(b) {
      a = d[j] - Hj * exp(b)
      if(a==-Inf) return(-1e5)
      if(a==Inf) return(1e5)
      return(a)
    }
    
   
    pred = X[j] * beta
    fim <- function(el) {
    #  print(el)
      a = el * fj(pred) - fj(pred + gamma_i * X[j]**2 * el * fj(pred))
      if(a < -1e100) return(-1e5)
      if(a > 1e100) return(1e5)
      # print("return")
      # print(a)
      return(a)
    }
#     lam0 = 0
#     lam1 = 0
#     fim0 = fim(lam0)
#     if(!is.na(fim0) && is.finite(fim0) && fim0 != 0 ) {
#       finished = F
#       while(!finished) {
#         lam1 = runif(1, min=-100, max=100)
#         fim1 = fim(lam1)
#        # print(sprintf("fim0=%.2f, lam1=%.2f fim1=%.2f", fim0, lam1, fim1))
#        if(is.infinite(fim1) || is.na(fim1)) 
#          finished <- F
#        else
#         finished <- (fim1 > 0 && fim0 < 0) || (fim1 < 0 && fim0 > 0)
#       }
#     }
#     
#     Bn = c(lam0, lam1)
#     print(Bn)
#     gamma_i = gammas[i]
#     lambda = NA
#     if(Bn[1]==Bn[2]) {
#       lambda = Bn[1] 
#     } else {
#       lambda = uniroot(f=fim, lower = Bn[1], upper=Bn[2], tol=1e-5)$root
#     #}
    lambda = optim(par=c(0), f = function(b) fim(b)**2, 
                   method = "L-BFGS")$par
    lams <- c(lams, fim(lambda))
    beta = beta + gamma_i * lambda * fj(pred) * X[j]
    betas.im = c(betas.im, beta)
  }
  par(mfrow=c(1, 2))
  plot(tail(betas.im, niters/2), type="l")
  print(sprintf("Last SGD = %.3f Last implicit = %.3f", tail(betas, 1), tail(betas.im, 1)))
  hist(lams, breaks=30)
print(summary(lams))
}
# test <- data.frame(start=runif(100000,1,100), stop=runif(100000,101,300), censor=round(runif(100000,0,1)), testfactor=round(runif(100000,1,11)))
# 
# test$testfactorf <- as.factor(test$testfactor)
# summ <- coxph(Surv(start,stop,censor) ~ relevel(testfactorf, 2), test)
