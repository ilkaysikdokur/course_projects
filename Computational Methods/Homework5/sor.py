import numpy as np

'''
Iteratively solves system of linear equations

Input:
n: how many variables (positive integer)
A: matrix of coefficients (double nxn matrix)
b: results of system (double n vector)
w: relaxation factor (positive double)

Numerical Method: SOR

Output:
x: approximated values of variables (double n vector)
'''

def SOR(n, A, b, w):
    kmax = 100
    delta = 1e-10
    epsilon = 1/2*1e-4
    
    x = np.zeros(n)
    y = np.zeros(n)
    
    print('k | x')
    
    for k in range(1, kmax+1):
        for i in range(n):
            y[i] = x[i]
        for i in range(0, n):
            sum = b[i]
            diag = A[i][i]
            if abs(diag) < delta:
                print('diagonal element too small')
                return
            for j in range(0, i):
                sum = sum - A[i][j]*x[j]
            for j in range(i+1, n):
                sum = sum - A[i][j]*x[j] 
            x[i] = sum / diag
            x[i] = w*x[i] + (1-w)*y[i]
        print(k, x)
        
        if np.linalg.norm(x-y) < epsilon:
            return
        
    print('maximum iterations reached')
    return