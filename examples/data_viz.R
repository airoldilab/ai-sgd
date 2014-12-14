## Functions to visualize the datasets we run experiments on.
#

mnist.plot.digit <- function(mnist.data, id) {
  # Plots the digit.
  # 
  # SYNOPSIS:
  #   d = read.csv("examples/data/..train.csv")
  #   mnist.plot.digit(d, 100)
  #
  pixels = mnist.data[id, -c(1)]
  stopifnot(length(pixels) == 28 * 28, all(is.element(pixels, 0:255)))
  x = c()
  y = c()
  colors = c()
  # Translate data to points and colors.
  for(point in seq(0, length(pixels)-1)) {
    j = point %% 28
    i = (point - j) / 28
    x = c(x, j)
    y = c(y, -i)
    colors = c(colors, gray(1-pixels[point + 1] / 255))
  }
  # Plot the digit.
  plot(x, y, col=colors, pch=16)
  print(sprintf("Digit is %d", mnist.data[id, 1]))
}