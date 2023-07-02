import bisection as bs
import math

nmax = 20
epsilon = (1e-6)/2

#CP 3.1.7
a = 0
b = 1
print ('Bisection: f(x) = x^3 + 3*x - 1, a = '+str(a)+', b = '+str(b)+', nmax = '+str(nmax)+', epsilon = '+str(epsilon))
bs.Bisection(lambda x: math.pow(x, 3) + 3*x - 1, a, b, 20, (1e-6)/2)
print ('\nFalse Position: f(x) = x^3 + 3*x - 1, a = '+str(a)+', b = '+str(b)+', nmax = '+str(nmax)+', epsilon = '+str(epsilon))
bs.False_Position(lambda x: math.pow(x, 3) + 3*x - 1, a, b, 20, (1e-6)/2)


a = 0.5
b = 2
print ('\n\nBisection: g(x) = x^3 - 2*sin(x), a = '+str(a)+', b = '+str(b)+', nmax = '+str(nmax)+', epsilon = '+str(epsilon))
bs.Bisection(lambda x: math.pow(x, 3) - 2*math.sin(x), a, b, 20, (1e-6)/2)
print ('\nFalse Position: g(x) = x^3 - 2*sin(x) a = '+str(a)+', b = '+str(b)+', nmax = '+str(nmax)+', epsilon = '+str(epsilon))
bs.False_Position(lambda x: math.pow(x, 3) - 2*math.sin(x), a, b, 20, (1e-6)/2)


a = 120
b = 130
print ('\n\nBisection: h(x) = x + 10 - x*cosh(50/x), a = '+str(a)+', b = '+str(b)+', nmax = '+str(nmax)+', epsilon = '+str(epsilon))
bs.Bisection(lambda x: x + 10 - x*math.cosh(50/x), a, b, 20, (1e-6)/2)
print ('\nFalse Position: h(x) = x^3 - 2*sin(x), a = '+str(a)+', b = '+str(b)+', nmax = '+str(nmax)+', epsilon = '+str(epsilon))
bs.False_Position(lambda x: x + 10 - x*math.cosh(50/x), a, b, 20, (1e-6)/2)



