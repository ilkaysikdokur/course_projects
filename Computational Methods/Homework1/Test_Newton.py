import newton as nw
import math


nmax = 20
epsilon = (1e-6)/2
delta = (1e-6)/2

#CP 3.2.1
x0 = 7
print('f(t) = tan(t) - t, x0 = '+str(x0)+', nmax = '+str(nmax)+', epsilon = '+str(epsilon)+', delta = '+str(delta))
nw.newton(lambda t: math.tan(t) - t, lambda t: (1/math.cos(t))**2 - 1, x0, nmax, epsilon, delta)

x0 = 2
print('\ng(t) = e^t - sqrt(t+9), x0 = '+str(x0)+', nmax = '+str(nmax)+', epsilon = '+str(epsilon)+', delta = '+str(delta))
nw.newton(lambda t: math.e**t - math.sqrt(t+9), lambda t: math.e**t - 1/(2*(t-9)**2), x0, nmax, epsilon, delta)


#CP 3.2.36
x0 = 2
m = 2
print('\nf(x) = x^2, x0 = '+str(x0)+', nmax = '+str(nmax)+', epsilon = '+str(epsilon)+', delta = '+str(delta)+', m = '+str(m))
nw.newton(lambda x: x**2, lambda x: 2*x, x0, nmax, epsilon, delta, m)

x0 = 1
print('\nf(x) = x + x^(4/3), x0 = '+str(x0)+', nmax = '+str(nmax)+', epsilon = '+str(epsilon)+', delta = '+str(delta))
nw.newton(lambda x: x + x**(4/3), lambda x: 1 + (4/3)*x**(1/3), x0, nmax, epsilon, delta)


x0 = 1
m = 2
print('\nf(x) = x + x^2*sin(2/x), x0 = '+str(x0)+', nmax = '+str(nmax)+', epsilon = '+str(epsilon)+', delta = '+str(delta)+', m = '+str(m))
nw.newton(lambda x: x + x**2*math.sin(2/x), lambda x: 1 + 2*x*math.sin(2/x) - 2*math.cos(2/x), x0, nmax, epsilon, delta, m)