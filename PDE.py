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


Nx=1000
Ny=1000 # y
hx=5 # step x
hy=1 # step y
k = 0.2 


x_points=np.arange(start= 0,stop = Nx + hx, step = hx)
y_points=np.arange(start = -500, stop = 500+hy, step = hy)
y_steps = int(math.floor(Ny + hy)/hy)
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



# make data
x = A[:, 0]
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)
plt.title("A0")
fig.savefig('plots/lineA0-bigger.png')

# make data
x = A[:, 1]
y = y_points

# plot
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0)
plt.title("A1")
fig.savefig('plots/lineA1-bigger.png')

# make data
x = A[:, 2]
y = y_points

# plot
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0)
plt.title("A2")
fig.savefig('plots/lineA2-bigger.png')

# make data
x = A[:, 3]
y = y_points

# plot
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0)
plt.title("A3")
fig.savefig('plots/lineA3-bigger.png')


################################################################################################################################################################

xvals = x_points[0:3]
yvals = y_points
zvals = A.real[:, 0:3]

heatmap, ax = plt.subplots()

im = ax.imshow(zvals,cmap='inferno',extent=[xvals[0],xvals[len(xvals)-1],yvals[0],yvals[len(yvals)-1]],interpolation='nearest',origin='lower',aspect='auto')

ax.set(xlabel='some x', ylabel='some y')
cbar = heatmap.colorbar(im)
plt.title("Heatmap A.real 0-3")

heatmap.savefig('plots/heatmap0-3-bigger.png')

#############################################################
xvals = x_points[0:5]
yvals = y_points
zvals = A.real[:, 0:5]

heatmap, ax = plt.subplots()

im = ax.imshow(zvals,cmap='inferno',extent=[xvals[0],xvals[len(xvals)-1],yvals[0],yvals[len(yvals)-1]],interpolation='nearest',origin='lower',aspect='auto')

ax.set(xlabel='some x', ylabel='some y')
cbar = heatmap.colorbar(im)
plt.title("Heatmap A.real 0-5")

heatmap.savefig('plots/heatmap0-5-bigger.png')

#############################################################
xvals = x_points[0:10]
yvals = y_points
zvals = A.real[:, 0:10]

heatmap, ax = plt.subplots()

im = ax.imshow(zvals,cmap='inferno',extent=[xvals[0],xvals[len(xvals)-1],yvals[0],yvals[len(yvals)-1]],interpolation='nearest',origin='lower',aspect='auto')

ax.set(xlabel='some x', ylabel='some y')
cbar = heatmap.colorbar(im)
plt.title("Heatmap A.real 0-10")

heatmap.savefig('plots/heatmap0-10-bigger.png')

###############################################################
xvals = x_points[0:3]
yvals = y_points
zvals = A.real[:, 0:3]

heatmap, ax = plt.subplots()

im = ax.imshow(zvals,cmap='inferno',extent=[xvals[0],xvals[len(xvals)-1],yvals[0],yvals[len(yvals)-1]],interpolation='nearest',origin='lower',aspect='auto')

ax.set(xlabel='some x', ylabel='some y')
cbar = heatmap.colorbar(im)
plt.title("Heatmap A.real 0-20")

heatmap.savefig('plots/heatmap0-20-bigger.png')

############################################################

xvals = x_points
yvals = y_points
zvals = A.real

heatmap, ax = plt.subplots()

im = ax.imshow(zvals,cmap='inferno',extent=[xvals[0],xvals[len(xvals)-1],yvals[0],yvals[len(yvals)-1]],interpolation='nearest',origin='lower',aspect='auto')

ax.set(xlabel='some x', ylabel='some y')
cbar = heatmap.colorbar(im)
plt.title("Heatmap A.real")
heatmap.savefig('plots/heatmap-bigger.png')


def convert_to_decibel(x):
    ''' Function takes a vector (matrix row) and transfores the quantity to decibels '''
    a0 = x[0]
    new = []
    for i in x:
        if cmath.log10(i/a0) == 0:
            new = new + [1]
        else:
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
heatmap.savefig('plots/decibels-bigger.png')


# true solution

omega = 1/k

Atilda = np.zeros((y_steps, x_steps), dtype=np.complex_)

for i in range(x_steps):
    for j in range(y_steps):
        x_i = float(x_points[i])
        y_j = float(y_points[j])
        Atilda[j, i] = cmath.exp(-y_j*y_j/(omega*omega*(1. + (2.*1j* x_i)/(k*omega*omega)))) * 1/cmath.sqrt(1.+(2.*1j*x_i)/(k*float(omega)*float(omega)))

        
        

xvals = x_points
yvals = y_points
zvals = Atilda.real

heatmap, ax = plt.subplots()

im = ax.imshow(zvals,cmap='inferno',extent=[xvals[0],xvals[len(xvals)-1],yvals[0],yvals[len(yvals)-1]],interpolation='nearest',origin='lower',aspect='auto')

ax.set(xlabel='some x', ylabel='some y')
cbar = heatmap.colorbar(im)
plt.title("Heatmap Atilda.real - analytical")
plt.show()
heatmap.savefig('plots/heatmap_tilda-bigger.png')


# compare results

# make data
x1 = A[:, 0]
x2 = Atilda[:, 0]
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(x1, y)
ax.plot(x2, y, linestyle = 'dashed')
plt.title("A0")
plt.legend(('numerical', 'analytical'))
plt.show()
fig.savefig('plots/lineA0a-compare-bigger.png')




# make data
x1 = A[:, 1]
x2 = Atilda[:, 1]
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(x1, y)
ax.plot(x2, y, linestyle = 'dashed')
plt.title("A1")
plt.legend(('numerical', 'analytical'))
plt.show()
fig.savefig('plots/lineA1-compare-bigger.png')



# make data
x1 = A[:, 5]
x2 = Atilda[:, 5]
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(x1, y)
ax.plot(x2, y, linestyle = 'dashed')
plt.title("A5")
plt.legend(('numerical', 'analytical'))
plt.show()
fig.savefig('plots/lineA5-compare-bigger.png')




# make data
x1 = A[:, 6]
x2 = Atilda[:, 6]
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(x1, y)
ax.plot(x2, y, linestyle = 'dashed')
plt.title("A6")
plt.legend(('numerical', 'analytical'))
plt.show()
fig.savefig('plots/lineA6-compare-bigger.png')



# make data
x1 = A[:, 8]
x2 = Atilda[:, 8]
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(x1, y)
ax.plot(x2, y, linestyle = 'dashed')
plt.title("A8")
plt.legend(('numerical', 'analytical'))
plt.show()
fig.savefig('plots/lineA8-compare-bigger.png')


# make data
x1 = A[:, 20]
x2 = Atilda[:, 20]
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(x1, y)
ax.plot(x2, y, linestyle = 'dashed')
plt.title("A20")
plt.legend(('numerical', 'analytical'))
plt.show()
fig.savefig('plots/lineA20-compare-bigger.png')


###########################################################################################################
# decibels
###########################################################################################################

Btilda =  np.zeros((y_steps, x_steps), dtype=np.complex_)

for i in range(0, y_steps):

    Btilda[i,:] = convert_to_decibel(Atilda[i, :])
    

    


# make data
x1 = B[:, 1]
x2 = Btilda[:, 1]
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(y, x1)
ax.plot(y, x2, linestyle = 'dashed')
plt.title("A1-compare-decibels")
plt.xlabel('y')
plt.ylabel('decibels')
plt.legend(('numerical', 'analytical'))
plt.show()
fig.savefig('plots/A1-compare-decibels-bigger.png')

# make data
x1 = B[:, 2]
x2 = Btilda[:, 2]
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(y, x1)
ax.plot(y, x2, linestyle = 'dashed')
plt.title("A2-compare-decibels")
plt.xlabel('y')
plt.ylabel('decibels')
plt.legend(('numerical', 'analytical'))
plt.show()
fig.savefig('plots/A2-compare-decibels-bigger.png')




# make data
x1 = B[:, 10]
x2 = Btilda[:, 10]
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(y, x1)
ax.plot(y, x2, linestyle = 'dashed')
plt.title("A10-compare-decibels")
plt.xlabel('y')
plt.ylabel('decibels')
plt.legend(('numerical', 'analytical'))
plt.show()
fig.savefig('plots/A10-compare-decibels-bigger.png')

# make data
x1 = B[:, 20]
x2 = Btilda[:, 20]
y = y_points

# plot
fig, ax = plt.subplots()

ax.plot(y, x1)
ax.plot(y, x2, linestyle = 'dashed')
plt.title("A20-compare-decibels")
plt.xlabel('y')
plt.ylabel('decibels')
plt.legend(('plots/numerical', 'analytical'))
plt.show()
fig.savefig('plots/A20-compare-decibels-bigger.png')