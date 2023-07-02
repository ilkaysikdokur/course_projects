'''
Calculates approximate integral of a function in an interval

Input:
f: function to approximate integral of (lambda)
a: lower boundary of the interval (double)
b: upper boundary of the interval (double)
n: amount of partitions in the interval (positive integer)

Method:
Lower and Upper Sums

Output:
average: average of lower and upper sums (double)
'''

def Sums(f, a, b, n):
    h = (b-a)/n
    sum = 0
    for i in range(n, 0, -1):
        x = a + i*h
        sum = sum + f(x)
    sum_lower = sum*h
    sum_upper = sum_lower + h*(f(a) - f(b))
    average = (sum_lower + sum_upper)/2
    
    return average