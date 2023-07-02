import numpy as np

#estimates first derivative of a function at a point
#Input:
#f: function to derivate (lambda)
#x: point to estimate the derivative of f (double)
#n: array size of Richardson extrapolation (positive integer)
#h: step size
#methods: Taylor Expansion, Richardson Extrapolation
#Output:
#d: Richardson extrapolation array (double array)
def Derivative(f, x, n, h):
    d = np.zeros((n + 1, n + 1))
    for i in range(0, n + 1):
        d[i][0] = (f(x + h) - f(x - h)) / (2 * h)
        for j in range(1, i + 1):
            d[i][j] = d[i][j - 1] + (d[i][j - 1] - d[i - 1][j - 1]) / (4 ** j - 1)
        h = h / 2
    return d
