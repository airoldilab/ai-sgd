## Functions to visualize the datasets we run experiments on.
#
#
mnist.plot.digit <- function(mnist.data, id) {
  # Plots the digit.
  # 
  # SYNOPSIS:
  #   d = read.csv("examples/data/..train.csv")
  #   mnist.plot.digit(d, 100)
  #
  print(sprintf("Checking data with id=%d", id))
  pixels = mnist.data[id, -c(1)]
  stopifnot(id <= nrow(mnist.data),
            length(pixels) == 28 * 28, 
            all(is.element(pixels, 0:255)))
  # Translate data to points and colors.
  loc = seq(0, length(pixels)-1)
  loc.x = loc %% 28
  loc.y = -(loc - loc.x) / 28
  colors = gray(1 - pixels/255)

  # Plot the digit.
  plot(loc.x, loc.y, col=colors, pch=16)
  print(sprintf("Digit is %d", mnist.data[id, 1]))
}

