'''
Calculates approximate integral of a function in an interval

Input:
f: function to approximate integral of (lambda)
a: lower boundary of the interval (double)
b: upper boundary of the interval (double)
n: amount of partitions in the interval (positive integer)

Method:
Romberg

Output:
r: resulting Romberg array (double array)
'''

import numpy as np

def Romberg(f, a, b, n):
    r = np.zeros((n+1, n+1))
    h = b - a
    r[0][0] = h/2*(f(a) + f(b))
    for i in range(1, n+1):
        h = h/2
        sum = 0
        for k in range(1, 2**i, 2):
            sum = sum + f(a + k*h)
        r[i][0] = 1/2*r[i-1][0] + sum*h
        for j in range(1, i+1):
            r[i][j] = r[i][j-1] + (r[i][j-1] - r[i-1][j-1])/(4**j-1)     
    return r