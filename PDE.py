# LIBRARY
# vector manipulation
import numpy as np
# math functions
import math 

# THIS IS FOR PLOTTING

import matplotlib.pyplot as plt # side-stepping mpl backend
import warnings
warnings.filterwarnings("ignore")



N=100
Nt=100 # y
h=5
k=5 # y
s = 0.2 ########### moj k
r = float(-1/2)/s ########## moj D

time=np.arange(start= 0,stop = Nt + k, step = k)
time_steps = int(math.floor(Nt + k - 0)/k)
x=np.arange(start = -50, stop = 50+h, step = h)

'''X, Y = np.meshgrid(x, time)
print(time)
print(x)
fig = plt.figure(figsize=(8,6))
plt.plot(X,Y,'ro');
plt.show();'''

########################3 s tukaj, je moj k

# initial condition
w=np.zeros((len(x),time_steps))
b=np.zeros(N-1)
# Initial Condition
for i in range (1,len(x)):
    w[i, 0] = 1

# Boundary Condition
for j in range (0,time_steps):
    w[0,j]= math.exp(-s*float(j^2))

'''print(w)
print(len(x))
print(len(w[:,0]))
fig = plt.figure(figsize=(8,4))
plt.plot(x,w[:,0],'o:',label='Initial Condition')
#plt.plot(x[[0,N]],w[[0,N],0],'go',label='Boundary Condition t[0]=0')
plt.title('Intitial and Boundary Condition',fontsize=24)
plt.xlabel('x')
plt.ylabel('w')
plt.legend(loc='best')
plt.show()'''

############################### r tukej je moj D

### moj initial value


aa = np.ones((3,3))
zz = np.ones((3,1)) ######## tp je pravo
mm = np.ones((1,3))
kk = np.ones(3)


'''print(np.multiply(aa,zz))
print(np.multiply(aa, mm))
print(np.multiply(aa, kk))'''

print(aa.dot(zz))
#print(aa.dot(mm))
print(aa.dot(kk))


A=np.zeros((N,N))
B=np.zeros((N,N))
for i in range (0,N):
    A[i,i]=2+2*r
    B[i,i]=2-2*r

for i in range (0,N-2):           
    A[i+1,i]=-r
    A[i,i+1]=-r
    B[i+1,i]=r
    B[i,i+1]=r
    
#print(len(A)) 
inverse = np.linalg.inv(A)
#print(inverse)
#print(A)
#print(B)
'''print(A)
print(w)
'''
'''Ainv=np.linalg.inv(A)   
fig = plt.figure(figsize=(12,4));
plt.subplot(121)
plt.imshow(A,interpolation='none');
plt.xticks(np.arange(N-1), np.arange(1,N-0.9,1));
plt.yticks(np.arange(N-1), np.arange(1,N-0.9,1));
clb=plt.colorbar();
clb.set_label('Matrix elements values');
plt.title('Matrix A r=%s'%(np.round(r,3)),fontsize=24)

plt.subplot(122)
plt.imshow(B,interpolation='none');
plt.xticks(np.arange(N-1), np.arange(1,N-0.9,1));
plt.yticks(np.arange(N-1), np.arange(1,N-0.9,1));
clb=plt.colorbar();
clb.set_label('Matrix elements values');
plt.title(r'Matrix $B$ r=%s'%(np.round(r,3)),fontsize=24)
fig.tight_layout()
plt.show();'''

'''for j in range (1,time_steps+1):
    b[0]=r*w[0,j-1]+r*w[0,j]
    b[N-2]=r*w[N,j-1]+r*w[N,j]
    v=np.dot(B,w[1:(N),j-1])
    w[1:(N),j]=np.dot(Ainv,v+b)'''

'''fig = plt.figure(figsize=(10,6));
plt.imshow(w.transpose(), aspect='auto')
plt.xticks(np.arange(len(x)), x)
plt.yticks(np.arange(len(time)), time)
plt.xlabel('x')
plt.ylabel('time')
clb=plt.colorbar()
clb.set_label('Temperature (w)')
plt.suptitle('Numerical Solution of the  Heat Equation r=%s'%(np.round(r,3)),fontsize=24,y=1.08)
fig.tight_layout()
plt.show()'''

initial = np.zeros((Nt, 1))
for i in range(Nt):
    initial[i,0] = math.exp(-s*s*float(i^2))

b = np.ones(Nt)*2*(1+r)
d = np.ones(Nt)*np.multiply(B, initial)
a = np.ones(Nt-1)*(-r)
c = np.ones(Nt-1)*(-r)
# lets create a, b, c, d
#print(d)



