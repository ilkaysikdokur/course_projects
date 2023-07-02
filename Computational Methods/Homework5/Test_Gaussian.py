import naive_gauss
import gauss
import solve
import numpy as np

n = 4
A = [[3,-13,9,3],[-6,4,1,-18],[6,-2,2,4],[12,-8,6,10]]
b = [-19,-34,16,26]

print('Solution of Ax = b system\n')

print('A: ')
print(np.array(A))
print('\nb: ')
print(np.array(b).reshape((1,n)).T)

#Naive Gaussian
print('\nNaive Gaussian Elimination: ')
x = naive_gauss.Naive_Gauss(n, A, b)
print('x:')
print(np.array(x).reshape((1,n)).T)

#Naive Gaussian
A = [[3,-13,9,3],[-6,4,1,-18],[6,-2,2,4],[12,-8,6,10]]
b = [-19,-34,16,26]

print('\nGaussian Elimination with Scaled Partial Pivoting: ')
A_scaled, l = gauss.Gauss(n, A)
x = solve.Solve(n, A_scaled, l, b)
print('x:')
print(np.array(x).reshape((1,n)).T)