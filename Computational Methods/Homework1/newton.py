#newton(f, f_drv, x, nmax, epsilon, delta, m=1)

#aim: it finds approximate value for roots of a function

#inputs:
#f: function that is aimed to find approximate roots of (lambda)
#f_drv: derivative of f (lambda)
#x: initial value for root approximation (double)
#nmax: maximum iteration amount (positive integer)
#epsilon: treshold for convergence control (positive double)
#delta: treshold for small derivative control (positive double)
#m: multiplicity of root, initial value -> 1 (positive integer)

#numerical method: Newton-Raphson method

#outputs:
#n: iteration amount (positive integer)
#x: approximated root in nth iteration (double)
#fx: f(x) in nth itaration (double)

def newton(f, f_drv, x, nmax, epsilon, delta, m=1):
    fx = f(x)
    print('n | x | f(x)')
    print(0, x, fx)
    for n in range(1, nmax+1):
        fp = f_drv(x)
        if abs(fp) < delta:
            print('small derivative')
            return
        d = fx/fp
        x = x-m*d
        fx = f(x)
        print(n, x, fx)
        if abs(d) < epsilon:
            print('convergence')
            return