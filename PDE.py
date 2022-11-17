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
    a0[y, 0] = math.exp(-k*k*(y*y))


s = [row[0] for row in a0]


a1 = []
for i in range(0, y_steps):
    y = y_points[i]
    a1.append(math.exp(-k*k*(y*y)))



'''x = a1
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)
plt.show()'''



# a, b, c, d for TDMA
D = np.zeros(y_steps, dtype=np.complex_)

for i in range (0,y_steps):
    D[i] = (-1/(2*(1j+1)*hy*hy))/k*hx  

a = np.ones(y_steps-1, dtype=np.complex_)*(-D[1:])
c = np.ones(y_steps-1, dtype=np.complex_)*(-D[:-1])
b = np.ones(y_steps, dtype=np.complex_)*2*(1+D)

A = np.zeros((y_steps, 1), dtype=np.complex_) # calculated a_n
for i in range(len(a0)):
    A[i, 0] = a0[i]



R = P.dot(A)

for x in range(1, x_steps+1):
    d = P.dot(A[:,x-1])
    x = np.linalg.solve(Q, d)
    s = x.reshape(y_steps, 1)
    A = np.hstack((A, s))

np.set_printoptions(precision=3)


#print(A[:, 0])

x = a1
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)
plt.title("A0 drawn real data type vector")
plt.show()
fig.savefig('lineA0-.png')

# make data
x = A[:, 0]
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)
plt.title("A0 drawn with i type data (0*i)")
plt.show()
fig.savefig('lineA0.png')

# make data
x = A[:, 1]
y = y_points

# plot
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0)
plt.title("A1")
plt.show()
fig.savefig('lineA1.png')

# make data
x = A[:, 2]
y = y_points

# plot
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0)
plt.title("A2")
plt.show()
fig.savefig('lineA2.png')

# make data
x = A[:, 3]
y = y_points

# plot
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0)
plt.title("A1")
plt.show()
fig.savefig('lineA3.png')

# make data
x = A[:, 2]
y = y_points

'''# plot
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)
plt.show()

xvals = x_points[0:3]
yvals = y_points
zvals = A[:, 0:3]

heatmap, ax = plt.subplots()

im = ax.imshow(zvals,cmap='inferno',extent=[xvals[0],xvals[len(xvals)-1],yvals[0],yvals[len(yvals)-1]],interpolation='nearest',origin='lower',aspect='auto')
ax.set(xlabel='some x', ylabel='some y')

cbar = heatmap.colorbar(im)
cbar.ax.set_ylabel('stuff')
plt.show()
heatmap.savefig('heatmap3p.png')'''

def huevalueplot(cmplxarray):
    # Creating the black cover layer

    black = np.full((*cmplxarray.shape, 4), 0.)
    black[:,:,-1] = np.abs(cmplxarray) / np.abs(cmplxarray).max()
    black[:,:,-1] = 1 - black[:,:,-1]

    # Actual plot

    fig, ax = plt.subplots()
    # Plotting phases using 'hsv' colormap (the 'hue' part)
    ax.imshow(np.angle(cmplxarray), cmap='hsv')
    # Plotting the modulus array as the 'value' part
    ax.imshow(black)
    ax.set_axis_off()
    plt.show()

