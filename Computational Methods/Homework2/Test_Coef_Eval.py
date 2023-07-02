import coef
import eval

x = []
y = []
a = []

n = 20
a0 = -5
an = 5
h = (an-a0)/n

for k in range(0, n+1):
    x.append(k*h)
    y.append(1/(x[k]**2+1))
    
a = coef.Coef(n, x, y)
e_max = 0
j_max = 0
t_max = 0
p_max = 0
f_max = 0

print('j | t | p | f | e')

for j in range(0, 41):
    t = j*h/2
    p = eval.Eval(n, x, a, t)
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