#calculates coefficients of Newton interpolation polynomial

#input:
#n: how many points are used for interpolation (positive integer)
#x: the points to build the interpolation on (double array)
#y: the image of the function to be interpolated at x (double array)

#method: Newton interpolation polynomial

#output:
#a: coefficients of the interpolation polynomial at each point (double array)
def Coef(n, x, y):
    a = []
    for i in range(0, n+1):
        a.append(y[i])
    for j in range(1, n+1):
        for i in range(n, j-1, -1):
            a[i] = (a[i] - a[i-1])/(x[i] - x[i-j])
    return a


