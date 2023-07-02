import inverse
import numpy as np

np.set_printoptions(suppress=True)

n = 4
A = [[-0.0001,5.096,5.101,1.853],[0,3.737,3.740,3.392],[0,0,0.006,5.254],[0,0,0,4.567]]

print('A: ')
print(np.array(A))

#Inverse
print('\nInverse of A: ')
A_inv = inverse.inverse(n,A)
print(np.array(A_inv))