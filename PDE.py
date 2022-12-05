# LIBRARY
import math
import numpy as np




'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------
PREPARE DATA
-------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

def save_vector_to_csv(x):
    pass


def calculate_h(x, y):
    ''' Calculates value of function h = h0 + tg(alpha) * y in point (x, y)'''

    alpha = 2
    alpha_rad = alpha*math.pi/180
    h0 = 50

    h = math.tan(alpha_rad) * y

    return h


def calculate_z(x,y, h):
    return h(x, y)

def calculate_K(f, c, h):
    
    omega = 2*math.pi*f
    len_h = len(h)

    # find maximum j
    min_h = min(h) # to get smallest bigest j
    M = math.floor(omega*h/(c*math.pi))

    # create empty matrix K
    K = np.zeros(len_h, 1)

    for j in range(M):
        Kj = np.zeros(len_h, 1)
        for i in len_h:
            Kj[i, 1] = math.sqrt(omega**2/c**2 - math.pi**2 * j**2 / h[i])

        K[:, j] = Kj

def calculate_phi(h, z):
    pass






