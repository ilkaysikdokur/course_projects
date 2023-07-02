import numpy as np

'''
Solves system of linear equations

Input:
n: how many variables (positive integer)
a: matrix of coefficients (double nxn matrix)
b: results of system (double n vector)

Numerical Method: LU Factorization

Output:
x: approximated values of variables (double n vector)
'''

def LU_Factorization(n, a, b):
    l = np.zeros((n,n))
    u = np.zeros((n,n))
    for k in range(0, n):
        l[k][k] = 1
        for j in range(k, n):
            sum = 0
            for s in range(0, k):
                sum = sum + l[k][s]*u[s][j]
            u[k][j] = a[k][j] - sum
        for i in range(k+1, n):
            sum = 0
            for s in range(0, k):
                sum = sum + l[i][s]*u[s][k]
            l[i][k] = (a[i][k] - sum) / u[k][k]
    
    z = np.zeros(n)
    z[0] = b[0]
    
    for i in range(1, n):
        sum = 0
        for j in range(0, i):
            sum = sum + l[i][j]*z[j]
        z[i] = b[i] - sum
        
    x = np.zeros(n)
    x[n-1] = z[n-1]/u[n-1][n-1]
    
    for i in range(n-2, -1, -1):
        sum = 0
        for j in range(i+1, n):
            sum = sum + u[i][j]*x[j]
        x[i] = (z[i] - sum) / u[i][i]
        
    return x