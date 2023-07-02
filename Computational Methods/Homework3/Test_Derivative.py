import math
import derivative as drv

n = 10
h = 1

x = 0
print('\nf(x) = cos(x), x = '+str(x)+', n = '+str(n)+', h = '+str(h))
d = drv.Derivative(lambda x: math.cos(x), x, n, h)
print('Estimated first derivative of cos(x) at '+str(x)+': '+str(d[n][n]))

x = 1
print('\nf(x) = atan(x), x = '+str(x)+', n = '+str(n)+', h = '+str(h))
d = drv.Derivative(lambda x: math.atan(x), x, n, h)
print('Estimated first derivative of atan(x) at '+str(x)+': '+str(d[n][n]))

x = 0
print('\nf(x) = abs(x), x = '+str(x)+', n = '+str(n)+', h = '+str(h))
d = drv.Derivative(lambda x: abs(x), x, n, h)
print('Estimated first derivative of abs(x) at '+str(x)+': '+str(d[n][n]))