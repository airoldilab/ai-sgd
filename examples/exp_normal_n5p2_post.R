#!/usr/bin/env Rscript
# Post-process the output obtained from running "exp_normal_n5_p2.R".
#
# @pre Current working directory is the root directory of this repository
# @pre Current working directory has the directory "img/"

library(dplyr)
library(ggplot2)

list.bias <- list()
# We stored only a subset of them.
subset.idx <- c(seq(100, 900, by=100), seq(1000, 1e5, by=1000))
for (i in 1:4) {
  # Load all data to plot.
  load(sprintf("out/exp_normal_n5p2_%i.RData", i))

  # Get method.
  if (i == 1) {
    method <- "SGD"
  } else if (i == 2) {
    method <- "I-SGD"
  } else if (i == 3) {
    method <- "AI-SGD"
  } else if (i == 4) {
    method <- "Batch"
  }

  # Save data frame of (t, estimated bias, method) triplets>
  list.bias[[i]] <- data.frame(
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
ggsave("img/exp_normal_n5p2.png")
