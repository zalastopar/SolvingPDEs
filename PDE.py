# LIBRARY
# vector manipulation
import numpy as np
# math functions
import math 
import cmath
# plot function
import matplotlib.pyplot as plt

# for finding errors
import traceback

# THIS IS FOR PLOTTING
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

 # side-stepping mpl backend
import warnings
warnings.filterwarnings("ignore")

'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------
CALCULATIONS FOR NUMERICAL SOLUTION
-------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

#----------------------------------------------------------------------------------------------------------------------
# Define area
#----------------------------------------------------------------------------------------------------------------------

Nx = 1000
Ny = 1000 # y
hx = 5 # step x
hy = 1 # step y
k = 0.2 

x_points = np.arange(start= 0,stop = Nx + hx, step = hx)
y_points = np.arange(start = -500, stop = 500+hy, step = hy)
y_steps = int(math.floor(Ny + hy)/hy) # number of y steps
x_steps = int(math.floor(Nx + hx)/hx) # number of x steps


#----------------------------------------------------------------------------------------------------------------------
# Prepare Q and P (Qa'=Pa)
#----------------------------------------------------------------------------------------------------------------------

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

#----------------------------------------------------------------------------------------------------------------------
# Initial condition a0
#----------------------------------------------------------------------------------------------------------------------

a0 = np.zeros((y_steps, 1), dtype=np.complex_)
for i in range(0, y_steps):
    y = y_points[i]
    a0[i, 0] = complex(math.exp(-k*k*(y*y)), 0)




'''D = np.zeros(y_steps, dtype=np.complex_)

for i in range (0,y_steps):
    D[i] = (-1/(2*(1j+1)*hy*hy))/k*hx  '''# ------------------------------------------------------------------------------------------------------ne rabim?

#----------------------------------------------------------------------------------------------------------------------
# Save initial values a0 to complex matrix A
#----------------------------------------------------------------------------------------------------------------------

A = np.zeros((y_steps, 1), dtype=np.complex_) # calculated a_n
for i in range(len(a0)):
    A[i, 0] = complex(a0[i], 0)



#----------------------------------------------------------------------------------------------------------------------
# Calculate matrix A = (a0 a1 a2 ...)
# Q*a(i) = P*a(i-1) = d
#----------------------------------------------------------------------------------------------------------------------

for x in range(1, x_steps+1):
    d = P.dot(A[:,x-1])
    x = np.linalg.solve(Q, d) # solve the system
    s = x.reshape(y_steps, 1) # transpose the result
    A = np.hstack((A, s)) # add a(i) to A

np.set_printoptions(precision=3) # round result in colsole on 3 digits


''' ------ Some useful functions -----------------------------------------------------------------------------------'''

#----------------------------------------------------------------------------------------------------------------------
# Function to calculate absolute value of a matrix
#----------------------------------------------------------------------------------------------------------------------


def absolute_value_element(x):
    ''' Returns absolute valuse of complex number'''
    return x.real**2 + x.imag**2

def absolute_value_matrix(X):
    ''' Applies function 'absolute_value_element' to all elements of a matrix X'''

    return np.array([list(map(absolute_value_element, x)) for x in X])



#----------------------------------------------------------------------------------------------------------------------
# Function to transform unit to decibels
#----------------------------------------------------------------------------------------------------------------------

def convert_to_decibel_element(x):
    ''' Transforms the quantity of compex element x to decibels'''

    return 20*np.log10(absolute_value_element(x))



def convert_to_decibel(X):
    ''' Applies function 'convert_to_decibel_element' to all elements of a matrix X'''

    return np.array([list(map(convert_to_decibel_element, x)) for x in X])



'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------
CALCULATIONS FOR ANALYTICAL SOLUTION
-------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


#----------------------------------------------------------------------------------------------------------------------
# Calculate analytical solution
#----------------------------------------------------------------------------------------------------------------------

omega = 1/k

A_ana = np.zeros((y_steps, x_steps), dtype=np.complex_)

for i in range(x_steps):
    for j in range(y_steps):
        x_i = float(x_points[i])
        y_j = float(y_points[j])
        A_ana[j, i] = cmath.exp(-y_j*y_j/(omega*omega*(1. + (2.*1j* x_i)/(k*omega*omega)))) * 1/cmath.sqrt(1.+(2.*1j*x_i)/(k*float(omega)*float(omega)))

 

'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------
PLOTS 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


#----------------------------------------------------------------------------------------------------------------------
# Plots of solutions with fixed x (Ax)
#----------------------------------------------------------------------------------------------------------------------


def plt_fixed_x(A, col, name = '', limits = 0):
    ''' Function plots one column of matrix A and 
    saves it as ('line-Ax' + name) in folder 'Plots' '''

    # choose the right row
    x = A[:, col]
    y = y_points

    # plot
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    plt.title('A' + str(col) + ' ' + name)
    plt.xlabel('A' + str(col))
    plt.ylabel('y points')
    if limits != 0:
        plt.ylim(limits)
    fig.savefig('plots/line-A' + str(col) + name + '.png')

# NUMERICAL ----------------------------------------
#---------------------------------------------------
# Plot all the columns in x_choices from A.real and abs(A)

x_choices = [0, 1, 2, 3]

for el in x_choices:
    plt_fixed_x(A.real, el, '_real_bigger', (-80, 80))                            # A.real
    plt_fixed_x(absolute_value_matrix(A), el, '_absolute_bigger', (-80, 80))      # abs(A)

# ANALYTICAL ----------------------------------------
#---------------------------------------------------
# Plot all the columns in x_choices from A.real and abs(A)



#----------------------------------------------------------------------------------------------------------------------
# Heatmap 
#----------------------------------------------------------------------------------------------------------------------

x_interval_choices = [[0, 3], [0, 5], [0, 10], [0, 20]]

def plt_heatmap_interval(A, b, e, name= ''):
    ''' Function plots a heatmap of matrix A from columns b to e and 
    saves it as ('Heatmap-Ab-e + name) in folder 'Plots' '''

    # choose the right rows
    xvals = x_points[b:e]
    yvals = y_points
    zvals = A[:, b:e]

    # plot
    heatmap, ax = plt.subplots()
    im = ax.imshow(zvals,cmap='inferno',extent=[xvals[0],xvals[len(xvals)-1],yvals[0],yvals[len(yvals)-1]],interpolation='nearest',origin='lower',aspect='auto')
    ax.set(xlabel='A[' + str(b)  + ', ' + str(e) + ']', ylabel='y points')
    cbar = heatmap.colorbar(im)
    plt.title('Heatmap' + ' A[' + str(b)  + ', ' + str(e) + '] ' + name)
    heatmap.savefig('plots/heatmap-A' + str(b) + '-' + str(e) + name + '.png')

def plt_heatmap(A, name = ''):
    ''' Function plots a heatmap of matrix A and 
    saves it as ('Heatmap-A + name) in folder 'Plots' '''

    # choose the right rows
    xvals = x_points
    yvals = y_points
    zvals = A

    # plot
    heatmap, ax = plt.subplots()
    im = ax.imshow(zvals,cmap='inferno',extent=[xvals[0],xvals[len(xvals)-1],yvals[0],yvals[len(yvals)-1]],interpolation='nearest',origin='lower',aspect='auto')
    ax.set(xlabel='A', ylabel='y points')
    cbar = heatmap.colorbar(im)
    plt.title('Heatmap-A' + name)
    heatmap.savefig('plots/heatmap-A' + name + '.png')


#---------------------------------------------------
# Plot all the intervals in x_interval_choices from A.real and abs(A)
for el in x_interval_choices:
    plt_heatmap_interval(A.real, el[0], el[1], '_real_bigger')                         # real
    plt_heatmap_interval(absolute_value_matrix(A), el[0], el[1], '_absolute_bigger')   # absolute


# NUMERICAL ---------------------------------------
#---------------------------------------------------
# Plot heatmap of matrix A.real and abs(A) and decibels(A)

plt_heatmap(A.real, '_real_num')                                   # real
plt_heatmap(absolute_value_matrix(A), '_absolute_num_bigger')      # absolute
plt_heatmap(convert_to_decibel(A), '_decibels_num_bigger')         # decibels


# ANALYTICAL ---------------------------------------
#---------------------------------------------------
# Plot heatmap of matrix A.real and abs(A) and decibels(A)

plt_heatmap(absolute_value_matrix(A_ana), '_absolute__ana_bigger')  # absolute
plt_heatmap(convert_to_decibel(A_ana), '_decibels_ana_bigger')      # decibels

  
#----------------------------------------------------------------------------------------------------------------------
# Compare results
#----------------------------------------------------------------------------------------------------------------------

def plt_compare_fixed_x(A, B, col, name = '', limits = 0):
    ''' Function plots one column of matrix A (numerical) and the same one from matrix B (analitical) and
    saves it as ('line-Ax' + name) in folder 'Plots' '''

    # choose the right row
    xA = A[:, col]
    xB = B[:, col]
    y = y_points

    # plot
    fig, ax = plt.subplots()
    ax.plot(xA, y)
    ax.plot(xB, y, linestyle = 'dashed')

    plt.title('column ' + str(col) + ' ' + name)
    plt.xlabel('A' + str(col))
    plt.ylabel('y points')
    if limits != 0:
        plt.ylim(limits)
    plt.legend(('numerical', 'analytical'))
    fig.savefig('plots/line-compare-A' + str(col) + name + '.png')


#---------------------------------------------------
# PCompare all the columns in x_choices

x_choices_compare = [0, 1, 5, 6, 8, 20]

for el in x_choices_compare:
    plt_compare_fixed_x(A.real, A_ana.real, el, '_real_bigger')                                             # real
    plt_compare_fixed_x(absolute_value_matrix(A), absolute_value_matrix(A_ana), el, '_absolute_bigger')     # absolute
    plt_compare_fixed_x(convert_to_decibel(A), convert_to_decibel(A_ana), el, '_real_bigger')               # decibels












    

    











