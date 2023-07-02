#returns the value of interpolating polynomial at a point

#input:
#n: how many points are used for interpolation (positive integer)
#x: the points to build the interpolation on (double array)
#a: coefficients of the interpolation polynomial (double array)
#t: the point to find the value of on the interpolation polynomial (double)

#method: Newton interpolation polynomial

#output:
#temp: the value of the point on the interpolation polynomial (double)
def Eval(n, x, a, t):
    temp = a[n]
    for i in range(n-1, -1, -1):
        temp = temp*(t-x[i]) + a[i]
    return temp
