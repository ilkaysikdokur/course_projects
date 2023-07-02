#interchange(a, b)

#aim: interchanges values of parameters

#inputs:
#a: first value (double)
#b: second value (double)

#outputs:
#a: first value after interchange (double)
#b: second value after interchange (double)

def interchange(a, b):
    x_tmp = a
    a = b
    b = x_tmp


#secant(f, x1, x2, epsilon, delta, maxf, x, ierr)

#aim: it finds approximate value for roots of a function

#inputs:
#f: function that is aimed to find approximate roots of (lambda)
#x1: first initial estimate (double)
#x2: second initial estimate (double)
#epsilon: treshold for difference between two iterates (positive double)
#delta: treshold for function value at estimate (positive double)
#maxf: bound for function evaluation (positive integer)


#numerical method: Secant method

#outputs:
#n_eval: function evaluation amount (positive integer)
#x1: approximated root in nth iteration (double)
#fx1: f(x1) in nth itaration (double)
#x: final estimate of solution (double)
#ierr: error flag for tolerance test (integer)

def secant(f, x1, x2, epsilon, delta, maxf):
    n_eval = 0
    fx1 = f(x1)
    fx2 = f(x2)
    n_eval = n_eval + 2
    if abs(fx1)>abs(fx2):
        interchange(x1, x2)
        interchange(fx1, fx2)
    print('n_eval | x1 | f(x1)')
    print(n_eval, x1, fx1)
    while n_eval < maxf:
        if abs(fx1) > abs(fx2):
            interchange(x1, x2)
            interchange(fx1, fx2)
        d = (x2-x1)/(fx2-fx1)
        x2 = x1
        fx2 = fx1
        d = d*fx1
        x1 = x1-d
        fx1 = f(x1)
        n_eval = n_eval + 1
        print(n_eval, x1, fx1)
        if abs(fx1) < delta:
            print('convergence')
            x = x1
            ierr = 0
            return x, ierr
        if abs(fx2-fx1) < epsilon:
            print('small difference')
            x = x1
            ierr = 0
            return x, ierr
        if n_eval == maxf:
            x = x1
            ierr = 1
            return x, ierr