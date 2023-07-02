import math
import second_derivative as sdrv

n = 10
h = 1

x = 0.5
print('f(x) = sin(x), x = '+str(x)+', n = '+str(n)+', h = '+str(h))
d2 = sdrv.Second_Derivative(lambda x: math.sin(x), x, n, h)
print('Estimated second derivative of sin(x) at '+str(x)+': '+str(d2[n][n]))

