import jacobi
import gauss_seidel
import sor
import numpy as np

n = 4
A = [[7,1,-1,2],[1,8,0,-2],[-1,0,4,-1],[2,-2,-1,6]]
b = [3,-5,4,-3]

print('Solution of Ax = b system\n')

print('A: ')
print(np.array(A))
print('\nb: ')
print(np.array(b).reshape((1,n)).T)

#Jacobi
print('\nJacobi: ')
jacobi.Jacobi(n, A, b)

#Jacobi
print('\nGauss-Seidel: ')
gauss_seidel.Gauss_Seidel(n, A, b)

#Jacobi
w = 1.1
print('\nSOR (relaxation factor: '+str(w)+'): ')
sor.SOR(n, A, b, w)