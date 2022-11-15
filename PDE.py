# LIBRARY
# vector manipulation
import numpy as np
# math functions
import math 

from numba import jit 

#############
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

# THIS IS FOR PLOTTING
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

import matplotlib.pyplot as plt # side-stepping mpl backend
import warnings
warnings.filterwarnings("ignore")


Nx=100
Ny=100 # y
hx=5 # step x
hy=1 # step y
k = 0.2 


x_points=np.arange(start= 0,stop = Nx + hx, step = hx)
y_points=np.arange(start = -50, stop = 50+hy, step = hy)
y_steps = int(math.floor(100 + hy)/hy)
x_steps = int(math.floor(Nx + hx)/hx)


'''def TDMASolve(a, b, c, d):
    # https://en.wikibooks.org/wiki/Algorithm_Implementation/Linear_Algebra/Tridiagonal_matrix_algorithm
    n = len(a)
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    xc = []
    for j in range(1, n):
        if(bc[j - 1] == 0):
            ier = 1
            return
        ac[j] = ac[j]/bc[j-1]
        bc[j] = bc[j] - ac[j]*cc[j-1]
    if(b[n-1] == 0):
        ier = 1
        return
    for j in range(1, n):
        dc[j] = dc[j] - ac[j]*dc[j-1]
    dc[n-1] = dc[n-1]/bc[n-1]
    for j in range(n-2, -1, -1):
        dc[j] = (dc[j] - cc[j]*dc[j+1])/bc[j]
    return dc
'''

# prepare Q and P
Q=np.zeros((y_steps,y_steps))
P=np.zeros((y_steps,y_steps))

for i in range (0,y_steps):
    D = float(-1/(2*(i+1)))/k*hx/hy/hy
    Q[i,i]=2+2*D
    P[i,i]=2-2*D

for i in range (0,y_steps-1):  
    D = float(-1/(2*(i+1)))/k*hx/hy/hy    
    Q[i+1,i]=-D
    Q[i,i+1]=-D
    P[i+1,i]=D
    P[i,i+1]=D


# initial condition a0
a0 = np.ones((y_steps, 1))
for i in range(0, y_steps):
    y = y_points[i]
    a0[i, 0] = math.exp(-k*k*float(y^2))


A = P.dot(a0) # calculated a_n

# a, b, c, d for TDMA
D = np.zeros(y_steps)

for i in range (0,y_steps):
    D[i] = float(-1/(2*(i+1)*hy*hy))/k*hx  

a = np.ones(y_steps-1)*(-D[1:])
c = np.ones(y_steps-1)*(-D[:-1])
b = np.ones(y_steps)*2*(1+D)


R = P.dot(A)

for x in range(1, x_steps+1):
    d = P.dot(A[:,x-1])
    x = np.linalg.solve(Q, d)

    s = x.reshape(y_steps, 1)
    A = np.hstack((A, s))

np.set_printoptions(precision=3)
print(A)

xvals = x_points[3:5]
yvals = y_points
zvals = A[:, 3:5]

heatmap, ax = plt.subplots()

im = ax.imshow(zvals,cmap='inferno',extent=[xvals[0],xvals[len(xvals)-1],yvals[0],yvals[len(yvals)-1]],interpolation='nearest',origin='lower',aspect='auto')
ax.set(xlabel='some x', ylabel='some y')

cbar = heatmap.colorbar(im)
cbar.ax.set_ylabel('stuff')
plt.show()
heatmap.savefig('heatmap3-5.png')