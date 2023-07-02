#sign(a)
#aim: returns sign of parameter a
#input:
#a: double
#output:
#-1(a>0), 0(a=0), 1(a>0)
#returns sign of 
def sign(a):
    sign = 0
    if a > 0:
        sign = 1
    elif a < 0:
        sign = -1
    return sign

#Bisection(f, a, b, nmax, epsilon)
#input:
#f: function to approximate roots of (lambda)
#a: left boundary of interval (double)
#b: right boundary of interval (double)
#nmax: maximum iteration amount (positive integer)
#epsilon: convergence treshold (positive double)
#numerical method : Bisection method
#output:
#n: iteration count
#c: approximated root
#fc: f(c)
#error: distance between c and boundaries of interval in nth iteration

def Bisection(f, a, b, nmax, epsilon):
    fa = f(a)
    fb = f(b)
    if sign(fa) == sign(fb):
        print('a | b | f(a) | f(b)')
        print(a, b, fa, fb)
        print('Function has some signs at a and b')
        return
    error = b - a
    print('n | c | f(c) | error')
    for n in range(1, nmax+1):
        error = error/2
        c = a + error
        fc = f(c)
        print(n, c, fc, error)
        if abs(error) < epsilon:
            print('convergence')
            return
        if sign(fa) != sign(fc):
            b = c
            fb = fc
        else:
            a = c
            fa = fc


#False_Position(f, a, b, nmax, epsilon)
#input:
#f: function to approximate roots of (lambda)
#a: left boundary of interval (double)
#b: right boundary of interval (double)
#nmax: maximum iteration amount (positive integer)
#epsilon: convergence treshold (positive double)
#numerical method : False Position 
#output:
#n: iteration count
#c: approximated root
#fc: f(c)
#error: distance between c and closest boundary of interval in nth iteration to c

def False_Position(f, a, b, nmax, epsilon):
    fa = f(a)
    fb = f(b)
    if sign(fa) == sign(fb):
        print('a | b | f(a) | f(b)')
        print(a, b, fa, fb)
        print('Function has some signs at a and b')
        return
    print('n | c | f(c) | error')
    for n in range(1, nmax+1):
        c = (a*fb - b*fa)/(fb - fa)
        error = abs(b-c) if abs(c-a) > abs(b-c) else abs(c-a)
        fc = f(c)
        print(n, c, fc, error)
        if abs(error) < epsilon:
            print('convergence')
            return
        if sign(fa) != sign(fc):
            b = c
            fb = fc
        else:
            a = c
            fa = fc





