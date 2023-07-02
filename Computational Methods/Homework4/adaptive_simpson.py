'''
Calculates approximate integral of a function in an interval

Input:
f: function to approximate integral of (lambda)
a: lower boundary of the interval (double)
b: upper boundary of the interval (double)
epsilon: convergence treshold (positive double)
level: current level of the computation (integer)
level_max: maximum level of the computation (positive integer)

Method:
Adaptive Simpson

Output:
simpson_result: result of the Adaptive Simpson algorithm (double)
'''

def Adaptive_Simpson(f, a, b, epsilon, level, level_max):
    level = level + 1
    h = b-a
    c = (a+b)/2
    one_simpson = h*(f(a) + 4*f(c) + f(b))/6
    d = (a+c)/2
    e = (c+b)/2
    two_simpson = h*(f(a) + 4*f(d) + 2*f(c) + 4*f(e) + f(b))/12
    
    if level >= level_max:
        simpson_result = two_simpson
        #print('maximum level reached')
    else:
        if abs(two_simpson - one_simpson) < 15*epsilon:
            simpson_result = two_simpson + (two_simpson - one_simpson)/15
        else:
            left_simpson = Adaptive_Simpson(f, a, c, epsilon/2, level, level_max)
            right_simpson = Adaptive_Simpson(f, c, b, epsilon/2, level, level_max)
            simpson_result = left_simpson + right_simpson
    
    return simpson_result