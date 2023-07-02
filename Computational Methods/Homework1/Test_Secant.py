import secant as sc
import newton as nw
import math

maxf = 20
epsilon = (1e-6)/2
delta = (1e-6)/2
nmax = 20

#CP 3.3.2
x1 = -0.5
x2 = 0
print('Secant: f(x) = e^x - 3x^2, x1 = '+str(x1)+', x2 = '+str(x2)+', epsilon = '+str(epsilon)+', delta = '+str(delta)+', maxf: '+str(maxf))
x, ierr = sc.secant(lambda x: math.exp(x) - 3*x**2, x1, x2, epsilon, delta, maxf)
print('\nApproximated x with secant: ', x)
print('Tolerance error flag: ', ierr)

x0 = 4
print('\n\nNewton: f(x) = e^x - 3x^2, x0 = '+str(x0)+', epsilon = '+str(epsilon)+', delta = '+str(delta)+', nmax: '+str(nmax))
nw.newton(lambda x: math.exp(x) - 3*x**2, lambda x: math.exp(x) - 6*x, x0, nmax, epsilon, delta)