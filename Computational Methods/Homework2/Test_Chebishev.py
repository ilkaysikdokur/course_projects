import coef
import eval
import math


n = 20
m = 41
a = -5
b = 5
h = (b-a)/n


x_Chebishev1 = []
y_Chebishev1 = []
coeff1 = []
sumOfError1 = 0


for i in range(0, n+1):
    x_Chebishev1.append(1/2*(a+b) + 1/2*(b-a)*math.cos(((2*i+1)/(2*n+2))*math.pi))
    y_Chebishev1.append(1/(x_Chebishev1[i]**2+1))
    
coeff1 = coef.Coef(n, x_Chebishev1, y_Chebishev1)
e_max1 = 0
j_max1 = 0
t_max1 = 0
p_max1 = 0
f_max1 = 0

print('Newton interpolation polynomial, f(x): 1/(x^2+1), n: '+str(n)+', m: '+str(m)+', a: '+str(a)+', b: '+str(b))
print('Chebishev points of the first kind:')
print('\nj | t | p | f | e')

for j in range(0, m):
    t = a + j*h/((m-1)/n)
    p = eval.Eval(n, x_Chebishev1, coeff1, t)
    f = 1/(t**2+1)
    e = abs(f - p)
    sumOfError1 += e
    
    print(j, t, p, f, e)
    if e > e_max1:
        j_max1 = j
        t_max1 = t
        p_max1 = p
        f_max1 = f
        e_max1 = e
print('\nMax error at:')
print(j_max1, t_max1, p_max1, f_max1, e_max1)
print('\nTotal error: ', sumOfError1)

##########################################################################################

x_Chebishev2 = []
y_Chebishev2 = []
coeff2 = []
sumOfError2 = 0

for i in range(0, n+1):
    x_Chebishev2.append(1/2*(a+b) + 1/2*(b-a)*math.cos(i/n*math.pi))
    y_Chebishev2.append(1/(x_Chebishev2[i]**2+1))
    
coeff2 = coef.Coef(n, x_Chebishev2, y_Chebishev2)
e_max2 = 0
j_max2 = 0
t_max2 = 0
p_max2 = 0
f_max2 = 0

print('\n\nNewton interpolation polynomial, f(x): 1/(x^2+1), n: '+str(n)+', m: '+str(m)+', a: '+str(a)+', b: '+str(b))
print('Chebishev points of the second kind:')
print('\nj | t | p | f | e')

for j in range(0, m):
    t = a + j*h/((m-1)/n)
    p = eval.Eval(n, x_Chebishev2, coeff2, t)
    f = 1/(t**2+1)
    e = abs(f - p)
    sumOfError2 += e
    
    print(j, t, p, f, e)
    if e > e_max2:
        j_max2 = j
        t_max2 = t
        p_max2 = p
        f_max2 = f
        e_max2 = e
print('\nMax error at:')
print(j_max2, t_max2, p_max2, f_max2, e_max2)
print('\nTotal error: ', sumOfError2)





