import numpy as np

#sample size
N = 4
#data
X = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
Y = np.array([1, -1, 1, -1])
#kernel matrix
Q = np.zeros((N, N))
#kernel function
K = lambda x1, x2: (np.dot(x1, x2) + 1)**2

for m in range(N):
    for n in range(N):
        Q[m][n] = Y[n] * Y[m] * K(X[n], X[m])

print('Kernel Matrix: ')        
print(Q)
print('\nEigenvalues of Kernel Matrix: ')   
print(np.linalg.eig(Q)[0])


A = np.array([[-9,1,-1,1,1],[1,-9,1,-1,-1],[-1,1,-9,1,1],[1,-1,1,-9,-1],[1,-1,1,-1,0]])
b = np.array([-1,-1,-1,-1,0])
x = np.linalg.solve(A, b)
print('x:')
print(x.reshape((1,5)).T)


#####################################################################################################

import numpy as np

#sample size
N = 4
#data
X = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
Y = np.array([1, -1, 1, -1])
#found alphas
alpha = [0.125,0.125,0.125,0.125]
#found Phi
sq2 = np.sqrt(2)
Phi = lambda x: np.array([1,x[0]**2,x[1]**2,sq2*x[0]*x[1],sq2*x[0],sq2*x[1]])

w = np.zeros(6)
for i in range(N):
    w += alpha[i]*Y[i]*Phi(X[i])
print('w:')
print(w.reshape((1,len(w))).T)    
    
b = 0
for i in range(N):
    if alpha[i] > 0:
        b = np.dot(w,Phi(X[i])) - Y[i]
        print('b: ', b)
    
    
    
    
    