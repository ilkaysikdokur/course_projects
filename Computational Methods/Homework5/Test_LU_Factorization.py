import lu_factorization
import numpy as np

n = 4
A = [[3,-13,9,3],[-6,4,1,-18],[6,-2,2,4],[12,-8,6,10]]
b = [-19,-34,16,26]

print('Solution of Ax = b system\n')

print('A: ')
print(np.array(A))
print('\nb: ')
print(np.array(b).reshape((1,n)).T)

#LU Factorization
print('\nLU Factorization: ')
x = lu_factorization.LU_Factorization(n, A, b)
print('x:')
print(np.array(x).reshape((1,n)).T)