# LIBRARY
# vector manipulation
import numpy as np
# math functions
import math 
import cmath
# plot function
import matplotlib.pyplot as plt
import cplot

from numba import jit 

#############
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

# THIS IS FOR PLOTTING
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

 # side-stepping mpl backend
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


# prepare Q and P
Q=np.zeros((y_steps,y_steps), dtype=np.complex_)
P=np.zeros((y_steps,y_steps), dtype=np.complex_)

for i in range (0,y_steps):
    D = (-1/(2*(1j+1)))/k*hx/hy/hy
    Q[i,i]= 2 + 2*D
    P[i,i]= 2 - 2*D

for i in range (0,y_steps-1):  
    D = (-1/(2*(1j+1)))/k*hx/hy/hy    
    Q[i+1,i]=-D
    Q[i,i+1]=-D
    P[i+1,i]=D
    P[i,i+1]=D


# initial condition a0
a0 = np.zeros((y_steps, 1), dtype=np.complex_)
for i in range(0, y_steps):
    y = y_points[i]
    a0[i, 0] = complex(math.exp(-k*k*(y*y)), 0)


s = [row[0] for row in a0]



# a, b, c, d for TDMA
D = np.zeros(y_steps, dtype=np.complex_)

for i in range (0,y_steps):
    D[i] = (-1/(2*(1j+1)*hy*hy))/k*hx  

a = np.ones(y_steps-1, dtype=np.complex_)*(-D[1:])
c = np.ones(y_steps-1, dtype=np.complex_)*(-D[:-1])
b = np.ones(y_steps, dtype=np.complex_)*2*(1+D)

A = np.zeros((y_steps, 1), dtype=np.complex_) # calculated a_n
for i in range(len(a0)):
    A[i, 0] = complex(a0[i], 0)



R = P.dot(A)

for x in range(1, x_steps+1):
    d = P.dot(A[:,x-1])
    x = np.linalg.solve(Q, d)
    s = x.reshape(y_steps, 1)
    A = np.hstack((A, s))

np.set_printoptions(precision=3)



'''# make data
x = A[:, 0]
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)
plt.title("A0")
fig.savefig('lineA0.png')

# make data
x = A[:, 1]
y = y_points

# plot
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0)
plt.title("A1")
fig.savefig('lineA1.png')

# make data
x = A[:, 2]
y = y_points

# plot
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0)
plt.title("A2")
fig.savefig('lineA2.png')

# make data
x = A[:, 3]
y = y_points

# plot
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0)
plt.title("A3")
fig.savefig('lineA3.png')

# make data
x = A[:, 2]
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)
'''

xvals = x_points
yvals = y_points
zvals = A.real

heatmap, ax = plt.subplots()

im = ax.imshow(zvals,cmap='inferno',extent=[xvals[0],xvals[len(xvals)-1],yvals[0],yvals[len(yvals)-1]],interpolation='nearest',origin='lower',aspect='auto')

ax.set(xlabel='some x', ylabel='some y')
cbar = heatmap.colorbar(im)
plt.title("Heatmap A.real")
plt.show()
heatmap.savefig('heatmap.png')


def convert_to_decibel(x):
    a0 = x[0]
    new = []
    for i in x:
        new = new +  [20*cmath.log10(i/a0)]
    return new


B = np.zeros((y_steps, x_steps+1), dtype=np.complex_)

for i in range(0, y_steps):

    B[i,:] = convert_to_decibel(A[i, :])


    


xvals = x_points
yvals = y_points
zvals = B.real

heatmap, ax = plt.subplots()

im = ax.imshow(zvals,cmap='inferno',extent=[xvals[0],xvals[len(xvals)-1],yvals[0],yvals[len(yvals)-1]],interpolation='nearest',origin='lower',aspect='auto')
ax.set(xlabel='some x', ylabel='some y')

cbar = heatmap.colorbar(im)
cbar.ax.set_ylabel('')
plt.title("Heatmap 20*log10(A0/A).real")
plt.show()
heatmap.savefig('decibels.png')


