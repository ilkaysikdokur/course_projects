import numpy as np

'''
Solves system of linear equations

Input:
n: how many variables (positive integer)
a: matrix of scaled coefficients (double nxn matrix)
l: index array (integer n vector) 
b: results of system (double n vector)

Numerical Method: Gaussian Elimination with Scaled Partial Pivoting

Output:
x: approximated values of variables (double n vector)
'''

def Solve(n, a, l, b):
    x = np.zeros(n)
    for k in range(0, n-1):
        for i in range(k+1, n):
            b[l[i]] = b[l[i]] - a[l[i]][k] * b[l[k]]
    x[n-1] = b[l[n-1]] / a[l[n-1]][n-1]
    for i in range(n-2, -1, -1):
        sum = b[l[i]]
        for j in range(i+1, n):
            sum = sum - a[l[i]][j] * x[j]
        x[i] = sum / a[l[i]][i]
    return x