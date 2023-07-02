'''
Calculates approximate integral of a function in an interval

Input:
f: function to approximate integral of (lambda)
a: lower boundary of the interval (double)
b: upper boundary of the interval (double)
n: amount of partitions in the interval (positive integer)
flag: flag for two-point or three-point algorithm (0 for two-point, 1 for three-point)

Method:
Composite Gaussian

Output:
sum: result of the Composite Gaussian algorithm (double)
'''

import math

def Composite_Gaussian(f, a, b, n, flag):
    n = n*2
    h = (b-a)/n
    sum = 0
    if flag == 0:
        #two-point
        A = [1, 1]
        x = [-1*math.sqrt(1/3), math.sqrt(1/3)]
        for j in range(1, int(n/2)+1):
            for i in range(2):
                sum = sum + A[i]*f(h*x[i] + (a + (2*j-1)*h))
    elif flag == 1:
        #three-point
        A = [5/9, 8/9, 5/9]
        x = [-1*math.sqrt(3/5), 0, math.sqrt(3/5)]
        for j in range(1, int(n/2)+1):
            for i in range(3):
                sum = sum + A[i]*f(h*x[i] + (a + (2*j-1)*h))
        
        
    sum = sum*h
    return sum