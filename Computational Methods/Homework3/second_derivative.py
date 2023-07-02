import numpy as np

#estimates second derivative of a function at a point
#Input:
#f: function to derivate (lambda)
#x: point to estimate the derivative of f (double)
#n: array size of Richardson extrapolation (positive integer)
#h: step size
#methods: Taylor Expansion, Richardson Extrapolation
#Output:
#d2: Richardson extrapolation array (double array)
def Second_Derivative(f, x, n, h):
    d2 = np.zeros((n+1, n+1))
    for i in range(0, n+1):
        d2[i][0] = (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2
        for j in range(1, i + 1):
            d2[i][j] = d2[i][j - 1] + (d2[i][j - 1] - d2[i - 1][j - 1]) / (4 ** j - 1)
        h = h / 2
    return d2

