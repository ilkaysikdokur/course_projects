'''
Calculates approximate integral of a function in an interval

Input:
f: function to approximate integral of (lambda)
a: lower boundary of the interval (double)
b: upper boundary of the interval (double)
n: amount of partitions in the interval (positive integer)

Method:
Composite Simpson

Output:
sum: result of the Composite Simpson algorithm (double)
'''

def Composite_Simpson(f, a, b, n):
    n = n*2
    h = (b-a)/n
    sum = f(a) + f(b)
    for i in range(1, int(n/2) + 1):
        sum = sum + 4*f(a + (2*i-1)*h)
    for k in range(1, int(n/2)):
        sum = sum + 2*f(a + 2*k*h)
    sum = sum*h/3
    return sum