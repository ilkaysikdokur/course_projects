import sums
import trapezoid
import romberg
import composite_simpson
import adaptive_simpson
import composite_gaussian

import math

#parameter for Romberg
romberg_n = 8

#parameters for Adaptive Simpson
epsilon = (1e-5)/2
level_max = 4

#Test function 1: f(x) = e^(-x)*cos(x)
f = lambda x: math.exp(-x)*math.cos(x)
a = 0
b = 2*math.pi
n = 20


print('f(x) = e^(-x)*cos(x)')
print('General Parameters: a = '+str(a)+', b = 2*pi, n = '+str(n))
print('Romberg: n = '+str(romberg_n))
print('Adaptive Simpson: epsilon = '+str(epsilon)+', level_max = '+str(level_max))

res_Sums = sums.Sums(f, a, b, n)
print('\nUpper and Lower Sums Method Result: '+str(res_Sums))
res_Trapezoid = trapezoid.Trapezoid(f, a, b, n)
print('Composite Trapezoid Method Result: '+str(res_Trapezoid))
res_Romberg = romberg.Romberg(f, a, b, romberg_n)[romberg_n][romberg_n]
print('Romberg Method Result: '+str(res_Romberg))
res_Composite_Simpson = composite_simpson.Composite_Simpson(f, a, b, n)
print('Composite Simpson Method Result: '+str(res_Composite_Simpson))
res_Adaptive_Simpson = adaptive_simpson.Adaptive_Simpson(f, a, b, epsilon, 0, level_max)
print('Adaptive Simpson Method Result: '+str(res_Adaptive_Simpson))
res_Composite_Gaussian_2pts = composite_gaussian.Composite_Gaussian(f, a, b, n, 0)
print('Composite Gaussian Method (two-points) Result: '+str(res_Composite_Gaussian_2pts))
res_Composite_Gaussian_3pts = composite_gaussian.Composite_Gaussian(f, a, b, n, 1)
print('Composite Gaussian Method (three-points) Result: '+str(res_Composite_Gaussian_3pts))

res_real = 0.499066278634146
print('\nReal Result: '+str(res_real))

print('\nUpper and Lower Sums Method Error: '+str(abs(res_Sums-res_real)))
print('Composite Trapezoid Method Error: '+str(abs(res_Trapezoid-res_real)))
print('Romberg Method Error: '+str(abs(res_Romberg-res_real)))
print('Composite Simpson Method Error: '+str(abs(res_Composite_Simpson-res_real)))
print('Adaptive Simpson Method Error: '+str(abs(res_Adaptive_Simpson-res_real)))
print('Composite Gaussian Method (two-points) Error: '+str(abs(res_Composite_Gaussian_2pts-res_real)))
print('Composite Gaussian Method (three-points) Error: '+str(abs(res_Composite_Gaussian_3pts-res_real)))


#Test function 2: f(x) = 1/(1+x^2)
f = lambda x: 1/(1+x**2)
a = 0
b = 1
n = 20

print('\nf(x) = 1/(1+x^2)')
print('General Parameters: a = '+str(a)+', b = '+str(b)+', n = '+str(n))
print('Romberg: n = '+str(romberg_n))
print('Adaptive Simpson: epsilon = '+str(epsilon)+', level_max = '+str(level_max))

res_Sums = sums.Sums(f, a, b, n)
print('\nUpper and Lower Sums Method Result: '+str(res_Sums))
res_Trapezoid = trapezoid.Trapezoid(f, a, b, n)
print('Composite Trapezoid Method Result: '+str(res_Trapezoid))
res_Romberg = romberg.Romberg(f, a, b, romberg_n)[romberg_n][romberg_n]
print('Romberg Method Result: '+str(res_Romberg))
res_Composite_Simpson = composite_simpson.Composite_Simpson(f, a, b, n)
print('Composite Simpson Method Result: '+str(res_Composite_Simpson))
res_Adaptive_Simpson = adaptive_simpson.Adaptive_Simpson(f, a, b, epsilon, 0, level_max)
print('Adaptive Simpson Method Result: '+str(res_Adaptive_Simpson))
res_Composite_Gaussian_2pts = composite_gaussian.Composite_Gaussian(f, a, b, n, 0)
print('Composite Gaussian Method (two-points) Result: '+str(res_Composite_Gaussian_2pts))
res_Composite_Gaussian_3pts = composite_gaussian.Composite_Gaussian(f, a, b, n, 1)
print('Composite Gaussian Method (three-points) Result: '+str(res_Composite_Gaussian_3pts))

res_real = 0.7853981633974483
print('\nReal Result: '+str(res_real))

print('\nUpper and Lower Sums Method Error: '+str(abs(res_Sums-res_real)))
print('Composite Trapezoid Method Error: '+str(abs(res_Trapezoid-res_real)))
print('Romberg Method Error: '+str(abs(res_Romberg-res_real)))
print('Composite Simpson Method Error: '+str(abs(res_Composite_Simpson-res_real)))
print('Adaptive Simpson Method Error: '+str(abs(res_Adaptive_Simpson-res_real)))
print('Composite Gaussian Method (two-points) Error: '+str(abs(res_Composite_Gaussian_2pts-res_real)))
print('Composite Gaussian Method (three-points) Error: '+str(abs(res_Composite_Gaussian_3pts-res_real)))


#Test function 3.1: f(x) = sin(x)
f = lambda x: math.sin(x)
a = 0
b = math.pi
n = 20

print('\nf(x) = sin(x)')
print('General Parameters: a = '+str(a)+', b = pi, n = '+str(n))
print('Romberg: n = '+str(romberg_n))
print('Adaptive Simpson: epsilon = '+str(epsilon)+', level_max = '+str(level_max))

res_Sums = sums.Sums(f, a, b, n)
print('\nUpper and Lower Sums Method Result: '+str(res_Sums))
res_Trapezoid = trapezoid.Trapezoid(f, a, b, n)
print('Composite Trapezoid Method Result: '+str(res_Trapezoid))
res_Romberg = romberg.Romberg(f, a, b, romberg_n)[romberg_n][romberg_n]
print('Romberg Method Result: '+str(res_Romberg))
res_Composite_Simpson = composite_simpson.Composite_Simpson(f, a, b, n)
print('Composite Simpson Method Result: '+str(res_Composite_Simpson))
res_Adaptive_Simpson = adaptive_simpson.Adaptive_Simpson(f, a, b, epsilon, 0, level_max)
print('Adaptive Simpson Method Result: '+str(res_Adaptive_Simpson))
res_Composite_Gaussian_2pts = composite_gaussian.Composite_Gaussian(f, a, b, n, 0)
print('Composite Gaussian Method (two-points) Result: '+str(res_Composite_Gaussian_2pts))
res_Composite_Gaussian_3pts = composite_gaussian.Composite_Gaussian(f, a, b, n, 1)
print('Composite Gaussian Method (three-points) Result: '+str(res_Composite_Gaussian_3pts))

res_real = 2
print('\nReal Result: '+str(res_real))

print('\nUpper and Lower Sums Method Error: '+str(abs(res_Sums-res_real)))
print('Composite Trapezoid Method Error: '+str(abs(res_Trapezoid-res_real)))
print('Romberg Method Error: '+str(abs(res_Romberg-res_real)))
print('Composite Simpson Method Error: '+str(abs(res_Composite_Simpson-res_real)))
print('Adaptive Simpson Method Error: '+str(abs(res_Adaptive_Simpson-res_real)))
print('Composite Gaussian Method (two-points) Error: '+str(abs(res_Composite_Gaussian_2pts-res_real)))
print('Composite Gaussian Method (three-points) Error: '+str(abs(res_Composite_Gaussian_3pts-res_real)))


#Test function 3.2: f(x) = e^x
f = lambda x: math.exp(x)
a = 0
b = 1
n = 20

print('\nf(x) = e^x')
print('General Parameters: a = '+str(a)+', b = '+str(b)+', n = '+str(n))
print('Romberg: n = '+str(romberg_n))
print('Adaptive Simpson: epsilon = '+str(epsilon)+', level_max = '+str(level_max))

res_Sums = sums.Sums(f, a, b, n)
print('\nUpper and Lower Sums Method Result: '+str(res_Sums))
res_Trapezoid = trapezoid.Trapezoid(f, a, b, n)
print('Composite Trapezoid Method Result: '+str(res_Trapezoid))
res_Romberg = romberg.Romberg(f, a, b, romberg_n)[romberg_n][romberg_n]
print('Romberg Method Result: '+str(res_Romberg))
res_Composite_Simpson = composite_simpson.Composite_Simpson(f, a, b, n)
print('Composite Simpson Method Result: '+str(res_Composite_Simpson))
res_Adaptive_Simpson = adaptive_simpson.Adaptive_Simpson(f, a, b, epsilon, 0, level_max)
print('Adaptive Simpson Method Result: '+str(res_Adaptive_Simpson))
res_Composite_Gaussian_2pts = composite_gaussian.Composite_Gaussian(f, a, b, n, 0)
print('Composite Gaussian Method (two-points) Result: '+str(res_Composite_Gaussian_2pts))
res_Composite_Gaussian_3pts = composite_gaussian.Composite_Gaussian(f, a, b, n, 1)
print('Composite Gaussian Method (three-points) Result: '+str(res_Composite_Gaussian_3pts))

res_real = 1.718281828459045
print('\nReal Result: '+str(res_real))

print('\nUpper and Lower Sums Method Error: '+str(abs(res_Sums-res_real)))
print('Composite Trapezoid Method Error: '+str(abs(res_Trapezoid-res_real)))
print('Romberg Method Error: '+str(abs(res_Romberg-res_real)))
print('Composite Simpson Method Error: '+str(abs(res_Composite_Simpson-res_real)))
print('Adaptive Simpson Method Error: '+str(abs(res_Adaptive_Simpson-res_real)))
print('Composite Gaussian Method (two-points) Error: '+str(abs(res_Composite_Gaussian_2pts-res_real)))
print('Composite Gaussian Method (three-points) Error: '+str(abs(res_Composite_Gaussian_3pts-res_real)))


#Test function 3.3: f(x) = arctan(x)
f = lambda x: math.atan(x)
a = 0
b = 1
n = 20

print('\nf(x) = arctan(x)')
print('General Parameters: a = '+str(a)+', b = '+str(b)+', n = '+str(n))
print('Romberg: n = '+str(romberg_n))
print('Adaptive Simpson: epsilon = '+str(epsilon)+', level_max = '+str(level_max))

res_Sums = sums.Sums(f, a, b, n)
print('\nUpper and Lower Sums Method Result: '+str(res_Sums))
res_Trapezoid = trapezoid.Trapezoid(f, a, b, n)
print('Composite Trapezoid Method Result: '+str(res_Trapezoid))
res_Romberg = romberg.Romberg(f, a, b, romberg_n)[romberg_n][romberg_n]
print('Romberg Method Result: '+str(res_Romberg))
res_Composite_Simpson = composite_simpson.Composite_Simpson(f, a, b, n)
print('Composite Simpson Method Result: '+str(res_Composite_Simpson))
res_Adaptive_Simpson = adaptive_simpson.Adaptive_Simpson(f, a, b, epsilon, 0, level_max)
print('Adaptive Simpson Method Result: '+str(res_Adaptive_Simpson))
res_Composite_Gaussian_2pts = composite_gaussian.Composite_Gaussian(f, a, b, n, 0)
print('Composite Gaussian Method (two-points) Result: '+str(res_Composite_Gaussian_2pts))
res_Composite_Gaussian_3pts = composite_gaussian.Composite_Gaussian(f, a, b, n, 1)
print('Composite Gaussian Method (three-points) Result: '+str(res_Composite_Gaussian_3pts))

res_real = 0.4388245731174757
print('\nReal Result: '+str(res_real))

print('\nUpper and Lower Sums Method Error: '+str(abs(res_Sums-res_real)))
print('Composite Trapezoid Method Error: '+str(abs(res_Trapezoid-res_real)))
print('Romberg Method Error: '+str(abs(res_Romberg-res_real)))
print('Composite Simpson Method Error: '+str(abs(res_Composite_Simpson-res_real)))
print('Adaptive Simpson Method Error: '+str(abs(res_Adaptive_Simpson-res_real)))
print('Composite Gaussian Method (two-points) Error: '+str(abs(res_Composite_Gaussian_2pts-res_real)))
print('Composite Gaussian Method (three-points) Error: '+str(abs(res_Composite_Gaussian_3pts-res_real)))