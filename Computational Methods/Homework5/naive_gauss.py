import numpy as np

'''
Solves system of linear equations

Input:
n: how many variables (positive integer)
a: matrix of coefficients (double nxn matrix)
b: results of system (double n vector)

Numerical Method: Naive Gaussian Elimination

Output:
x: approximated values of variables (double n vector)
'''

def Naive_Gauss(n, a, b):
    x = np.zeros(n)
    for k in range(0, n-1):
        for i in range(k+1, n):
            xmult = a[i][k] / a[k][k]
            a[i][k] = xmult
            for j in range(k+1, n):
                a[i][j] = a[i][j] - xmult * a[k][j]
            b[i] = b[i] - xmult * b[k]
    x[n-1] = b[n-1] / a[n-1][n-1]
    for i in range(n-2, -1, -1):    
        sum = b[i]
        for j in range(i+1, n):
            sum = sum - a[i][j] * x[j]
        x[i] = sum / a[i][i]
    return x
