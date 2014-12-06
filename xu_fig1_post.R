#!/usr/bin/env Rscript
# Replicate Figure 1 of Xu.

library(dplyr)
library(ggplot2)

list.bias <- list()
# We stored only a subset of them.
subset.idx <- c(seq(100, 900, by=100), seq(1000, 1e5, by=1000))
for (i in 1:9) {
  # Load all data to plot.
  load(sprintf("out/xu_fig1_%i.RData", i))

  # Get method.
  if (i == 1) {
    method <- "E-SGD"
  } else if (i == 2) {
    method <- "AE-SGD"
  } else if (i == 3) {
    method <- "LSE-SGD"
  } else if (i == 4) {
    method <- "LSAE-SGD"
  } else if (i == 5) {
    method <- "I-SGD"
  } else if (i == 6) {
    method <- "AI-SGD"
  } else if (i == 7) {
    method <- "LSI-SGD"
  } else if (i == 8) {
    method <- "LSAI-SGD"
  } else if (i == 9) {
    method <- "Batch"
  }

  # Save data frame of (t, estimated bias, method) triplets>
  list.bias[[i]] = data.frame(
    t=subset.idx,
    est.bias=apply(theta, 2, function(colum) {
      t(colum) %*% d$A %*% colum
      }),
    method=method
    )
}

# Create a data frame row binding each p x niters matrix for ggplot.
dat <- do.call(rbind, list.bias)

# Plot.
dat %>%
  ggplot(aes(x=t, y=est.bias, group=method, color=method)) +
    geom_line() +
    scale_x_log10(limits=c(1e2, 1e5), breaks=10^(2:6)) +
    scale_y_log10(limits=c(1e-4, 1e-1), breaks=10^(-6:2)) +
    xlab("Training size t") +
    ylab("Excess risk") +
    ggtitle("Excess risk over training size")
ggsave("img/xu_fig1.png")
