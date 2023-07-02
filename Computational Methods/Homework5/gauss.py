import numpy as np

'''
Swaps the values of two elements

Input:
a: first element (double)
b: second element (double)

Output:
b: swapped first element (double)
a: swapped second element (double)
'''
def swap(a, b):
    return b, a


'''
Scales coefficients and finds pivotal indexes

Input:
n: how many variables (positive integer)
a: matrix of coefficients (double nxn matrix)

Numerical Method: Gaussian Elimination with Scaled Partial Pivoting

Output:
a: scaled coefficients (double nxn matrix)
l: index array (integer n vector) 
'''

def Gauss(n, a):
    s = np.zeros(n)
    l = np.zeros(n).astype(np.int32)
    for i in range(0, n):
        l[i] = i
        smax = 0
        for j in range(0, n):
            smax = max(smax, abs(a[i][j]))
        s[i] = smax
    for k in range(0, n-1):
        rmax = 0
        for i in range(k, n):
            r = abs(a[l[i]][k] / s[l[i]])
            if r > rmax:
                rmax = r
                j = i
        l[k], l[j] = swap(l[k], l[j])
        for i in range(k+1, n):
            xmult = a[l[i]][k] / a[l[k]][k]
            a[l[i]][k] = xmult
            for j in range(k+1, n):
                a[l[i]][j] = a[l[i]][j] - xmult * a[l[k]][j]
    return a, l


