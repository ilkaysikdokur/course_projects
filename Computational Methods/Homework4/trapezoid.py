'''
Calculates approximate integral of a function in an interval

Input:
f: function to approximate integral of (lambda)
a: lower boundary of the interval (double)
b: upper boundary of the interval (double)
n: amount of partitions in the interval (positive integer)

Method:
Composite Trapezoid

Output:
sum: result of the Composite Trapezoid algorithm (double)
'''

def Trapezoid(f, a, b, n):
    h = (b-a)/n
    sum = (f(a) + f(b))/2
    for i in range(1, n):
        x = a + i*h
        sum = sum + f(x)
    sum = sum*h
    
    return sum
