import coef
import eval


n = 20
m = 41
a = -5
b = 5
h = (b-a)/n

x = []
y = []
ax = []


for k in range(0, n+1):
    x.append(a + k*h)
    y.append(1/(x[k]**2+1))
    
coeff = coef.Coef(n, x, y)
e_max = 0
j_max = 0
t_max = 0
p_max = 0
f_max = 0

print('Newton interpolation polynomial, f(x): 1/(x^2+1), n: '+str(n)+', m: '+str(m)+', a: '+str(a)+', b: '+str(b))
print('Equal points:')
print('\nj | t | p | f | e')

for j in range(0, m):
    t = a + j*h/((m-1)/n)
    p = eval.Eval(n, x, coeff, t)
    f = 1/(t**2+1)
    e = abs(f - p)
    print(j, t, p, f, e)
    if e > e_max:
        j_max = j
        t_max = t
        p_max = p
        f_max = f
        e_max = e
print('\nMax error at:')
print(j_max, t_max, p_max, f_max, e_max)