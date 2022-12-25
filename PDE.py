# LIBRARY
import math
import numpy as np
import pickle
import cmath

# plot function
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

bigger = False

'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------
PREPARE DATA
-------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
#----------------------------------------------------------------------------------------------------------------------
# Define area
#----------------------------------------------------------------------------------------------------------------------

Nx = 100
Ny = 100 # y
hx = 5 # step x
hy = 1 # step y
k = 0.2 
bigger = False

x_points = np.arange(start= 0,stop = Nx + hx, step = hx)
y_points = np.arange(start = -50, stop = 50+hy, step = hy)
y_steps = int(math.floor(Ny + hy)/hy) # number of y steps
x_steps = int(math.floor(Nx + hx)/hx) # number of x steps







def save_data(x, name):
    ''' Takes n array and saves it as 'name' in file 'data' as .pkl file'''

    with open('data/' + name + ".pkl","wb") as f:
        pickle.dump(x,f)

def load_data(name):
    ''' Takes data 'name' in file 'data' and returns n array that is saved in the data'''

    with open('data/' + name + ".pkl","rb") as f:
        x = pickle.load(f)

    return x


def calculate_h_x_y(x, y):
    ''' Calculates value of function h = h0 + tg(alpha) * y in point (x, y)'''

    alpha = 2
    alpha_rad = alpha*math.pi/180
    h0 = 50

    h = math.tan(alpha_rad) * y

    return h

def calculate_h(x, y):
    ''' calculates h(x, y) for every x, y (vectors)'''
    
    return np.array([[calculate_h_x_y(x_el, y_el) for y_el in y] for x_el in x])




def calculate_Kj(omega, c, x, y, j, h):
    ''' 
    x, y vectors, j float, omega float, c float
    h is matrix of h(x, y) for all x, y
    returns matrix Kj
    '''

    # create empty matrix Kj
    K = np.zeros((len(y), len(x)), dtype=np.complex_)

    #########h = calculate_h(x, y)

    for i in range(len(y)):
        for m in range(len(x)):
            K[i, m] = cmath.sqrt(omega**2/c**2 - (cmath.pi*j/h[m, i])**2)


    return np.transpose(K)


def calculate_K00j(omega, c, j):
    h = calculate_h_x_y(0, 0)
    if h == 0:
        h = 0.00000001
    return cmath.sqrt(omega**2/c**2 - (cmath.pi*j/h)**2)

def calculate_phi_j(z, x, y, j, h):
    '''
    function h, float z
    x, y, vectors
    h is matrix of h(x, y) for all x, y
    returns matrix phi_j for one z
    '''

    # create empty matrix Kj
    P = np.zeros((len(y), len(x)), dtype=np.complex_)

    for i in range(len(y)):
        for m in range(len(x)):
            P[i, m] = cmath.sqrt(2/h)* cmath.sin(math.pi*j*z/h[m, i])

    return P


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
SOLVE PDE
-------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def solvePDE_j(a0, Kj, Kj0, dx, dy, x_steps, y_steps):
    '''

    solves PDE 2iKj0 * Aj_x + Aj_yy + (Kj^2 - Kj0^2)Aj = 0
    returns Aj
    '''


    # calculate Kj^2 - Kj0^2

    Kj2 = np.square(Kj)
    K = Kj2 - Kj0**2

    # calculate D
    #Kj0 = 0.2 #######################
    D = - dx/(4*1j*Kj0*dy**2)
    

    # Prepare Q and P and f (Qa'=Pa + fa)

    f = dx*(-1/2*1j*Kj0)*K



    Q=np.zeros((y_steps,y_steps), dtype=np.complex_) 
    P=np.zeros((y_steps,y_steps), dtype=np.complex_)



    for i in range (0,y_steps):
        if i == 0 or i == (y_steps - 1):
            Q[i,i]= 1 + 1*D
            P[i,i]= 1 - 1*D #+ f[i, i]
        else:
            Q[i,i]= 1 + 2*D
            P[i,i]= 1 - 2*D #+ f[i, i]

    for i in range (0,y_steps-1):    
        Q[i+1,i]=-D
        Q[i,i+1]=-D
        P[i+1,i]=D
        P[i,i+1]=D



    # Save initial values a0 to complex matrix A

    A = np.zeros((y_steps, 1), dtype=np.complex_) # calculated a_n
    for i in range(len(a0)):
        A[i, 0] = complex(a0[i], 0)



    for x in range(1, x_steps+1):

        d = (P).dot(A[:,x-1])

        # update d
        diag = np.diag(np.matrix(f[x-1, :]))
        d_update = d + diag

        x = np.linalg.solve(Q, d_update)
        #x = np.linalg.solve(Q, d) # solve the system
        s = x.reshape(y_steps, 1) # transpose the result
        A = np.hstack((A, s)) # add a(i) to A

    return A



'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------
PLOTS 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------'''


#----------------------------------------------------------------------------------------------------------------------
# Plots of solutions with fixed x (Ax)
#----------------------------------------------------------------------------------------------------------------------


def plt_fixed_x(A, col, name = '', limits = 0):
    ''' Function plots one column of matrix A and 
    saves it as ('line-Ax' + name) in folder 'Plots' '''
    if bigger == True:
        name = name + '_bigger'

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
    fig.savefig('plots11/line-A' + str(col) + name + '.png')



#----------------------------------------------------------------------------------------------------------------------
# Heatmap 
#----------------------------------------------------------------------------------------------------------------------



def plt_heatmap_interval(A, b, e, name= ''):
    ''' Function plots a heatmap of matrix A from columns b to e and 
    saves it as ('Heatmap-Ab-e + name) in folder 'Plots' '''

    if bigger == True:
        name = name + '_bigger'

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
    heatmap.savefig('plots11/heatmap-A' + str(b) + '-' + str(e) + name + '.png')

def plt_heatmap(A, name = ''):
    ''' Function plots a heatmap of matrix A and 
    saves it as ('Heatmap-A + name) in folder 'Plots' '''

    if bigger == True:
        name = name + '_bigger'

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
    heatmap.savefig('plots11/heatmap-A' + name + '.png')



#----------------------------------------------------------------------------------------------------------------------
# Compare results
#----------------------------------------------------------------------------------------------------------------------

def plt_compare_fixed_x(A, B, col, name = '', limits = 0):
    ''' Function plots one column of matrix A (numerical) and the same one from matrix B (analitical) and
    saves it as ('line-Ax' + name) in folder 'Plots' '''

    if bigger == True:
        name = name + '_bigger'

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
    fig.savefig('plots11/line-compare-A' + str(col) + name + '.png')




'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------
PROGRAMME
-------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def funkcija():
    # values
    f = 100
    omega = 2*math.pi*f
    c = 1500


    Nx = 100
    Ny = 100 # y
    dx = 5 # step x
    dy = 1 # step y
    k = 0.2 
    bigger = False

    x_points = np.arange(start= 0,stop = Nx + hx, step = hx)
    y_points = np.arange(start = -50, stop = 50+hy, step = hy)
    y_steps = int(math.floor(Ny + hy)/hy) # number of y steps
    x_steps = int(math.floor(Nx + hx)/hx) # number of x steps


    # Initial condition a0
    a0 = np.zeros((y_steps, 1), dtype=np.complex_)
    for i in range(0, y_steps):
        y = y_points[i]
        a0[i, 0] = complex(math.exp(-k**2*y**2), 0)

    # calculate H(x,y)
    H = calculate_h(x_points, y_points)
    # where H is 0?????????????????????????????????????????'
    H[H==0.0] = 0.00000001


    # J
    for j in range(4):

        # calculate Kj
        Kj = calculate_Kj(omega, c, x_points, y_points, j, H)
        #Kj = np.zeros((y_steps,y_steps), dtype=np.complex_) 

        # calculate_K00
        Kj0 = calculate_K00j(omega, c, j)
        #Kj0 = 0

        A = solvePDE_j(a0, Kj, Kj0, dx, dy, x_steps, y_steps)


        # NUMERICAL ----------------------------------------
        #---------------------------------------------------
        # Plot all the columns in x_choices from A.real and abs(A)

        x_choices = [0, 1, 2, 3]

        for el in x_choices:
            #plt_fixed_x(A.real, el, '_real', (-80, 80))                            # A.real
            plt_fixed_x(absolute_value_matrix(A), el, '_absolute_J=' + str(j), (-80, 80))      # abs(A)


        #---------------------------------------------------
        # Plot all the intervals in x_interval_choices from A.real and abs(A)
        x_interval_choices = [[0, 3], [0, 5], [0, 10], [0, 20]]
        for el in x_interval_choices:
            #plt_heatmap_interval(A.real, el[0], el[1], '_real')                         # real
            plt_heatmap_interval(absolute_value_matrix(A), el[0], el[1], '_absolute_J=' + str(j))   # absolute


        # NUMERICAL ---------------------------------------
        #---------------------------------------------------
        # Plot heatmap of matrix A.real and abs(A) and decibels(A)

        #plt_heatmap(A.real, '_real_num')                                   # real
        plt_heatmap(absolute_value_matrix(A), '_absolute_num_J=' + str(j))      # absolute
        plt_heatmap(convert_to_decibel(A), '_decibels_num_J=' + str(j))         # decibels



funkcija()
