import gauss
import solve
import numpy as np

'''
Calculates inverse of a square matrix

Input:
n: dimension (positive integer)
A: square matrix (double nxn matrix)

Numerical Method: Gaussian Elimination with Scaled Partial Pivoting

Output:
A_inv: inverse of A(double nxn matrix)
'''

def inverse(n, A):
    A_inv = []
    for i in range(n):
        A_scaled = A
        b = np.zeros(n)
        b[i] = 1
        A_scaled, l = gauss.Gauss(n, A_scaled)
        x_i = solve.Solve(n, A_scaled, l, b)
        A_inv.append(x_i)
    A_inv = np.array(A_inv).T
    return A_inv