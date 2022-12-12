# LIBRARY
import math
import numpy as np
import pickle



'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------
PREPARE DATA
-------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def save_data(x, name):
    ''' Takes n array and saves it as 'name' in file 'data' as .pkl file'''

    with open('data/' + name + ".pkl","wb") as f:
        pickle.dump(x,f)

def load_data(name):
    ''' Takes data 'name' in file 'data' and returns n array that is saved in the data'''

    with open('data/' + name + ".pkl","rb") as f:
        x = pickle.load(f)

    return x


def calculate_h(x, y):
    ''' Calculates value of function h = h0 + tg(alpha) * y in point (x, y)'''

    alpha = 2
    alpha_rad = alpha*math.pi/180
    h0 = 50

    h = math.tan(alpha_rad) * y

    return h


def calculate_Kj(omega, c, x, y, j):
    ''' 
    x, y vectors, j float, omega float, c float
    returns matrix Kj
    '''

    # create empty matrix Kj
    K = np.zeros(len(y), len(x))

    for i in range(len(y)):
        for m in range(len(x)):
            h = calculate_h(x[m], y[i])
            K[i, m] = math.sqrt(omega**2/c**2 - (math.pi*j/h)**2)

    return K



def calculate_phi_j(h, z, x, y):
    '''
    function h, float z
    x, y, vectors
    returns matrix phi_j for one z
    '''

    # create empty matrix Kj
    P = np.zeros(len(y), len(x))

    for i in range(len(y)):
        for m in range(len(x)):
            h = calculate_h(x[m], y[i])
            P[i, m] = math.sqrt(2/h)* math.sin(math.pi*j*z/h)

    return P

