#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:16:21 2023

@author: alina
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:35:10 2022

@author: alina
"""

#------------------------------------------------------------------------------
#-----------------------------Atom's vacancies analysis------------------------
#------------------------------------------------------------------------------

'''
This program finds the equation describing the surface made by atoms. 
SymbolicRegression is used to train the model on the surface data to find the
appropriate equation. 
'''

# Import libraries
#------------------------------------------------------------------------------

from gplearn.genetic import SymbolicRegressor

from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d


import numpy as np
import graphviz

from gplearn.functions import *
#import readSqGr

# Define a function to count number of rows in a file
#------------------------------------------------------------------------------
# https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python
def rawincount(filename):
    
    from itertools import (takewhile,repeat)
    f = open(filename, 'rb')    
    bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))    
    numOfRows=sum( buf.count(b'\n') for buf in bufgen )
    f.close()
    
    return numOfRows

# Read data from the file and put them in arrays
#------------------------------------------------------------------------------

fname='stat-splot.dat'

nOfr=rawincount(fname)

print('rows:', nOfr)

dSize=nOfr/21


mx0=np.ndarray((21,7), float)
mx1=np.ndarray(mx0.shape, )
yd=np.ndarray(mx0.shape, )

fdat=open(fname, 'r')

i,j=0,0

for line in fdat:
    x,y,z=line.split()
    
    mx0[i,j]=float(x)/10
    mx1[i,j]=float(y)
    yd[i,j]=float(z)
    
    i+=1
    
    if i==21:
        i=0
        j+=1

fdat.close()    


# Prepare data the training dataset and functions needed for training 
#------------------------------------------------------------------------------

X_train=np.concatenate((mx0.reshape(147,1), mx1.reshape(147,1)), axis=1)
y_train=yd.reshape(147,1)

def _sqr(x):
    return x**2

def _cub(x):
    return x*x*x

def _prot_sqrt(x):    
    #with np.errstate(invalid='ignore'):
    return np.sqrt(np.abs(x))

     
sqr = make_function(function=_sqr, name='sqr', arity=1)
cub=make_function(function=_cub, name='cub', arity=1)
sqrtt=make_function(function=_prot_sqrt, name='sqrtt', arity=1)

# Define SymbolicRegressor and fit the data
#------------------------------------------------------------------------------

est_gp = SymbolicRegressor(population_size=500,
                           generations=100, stopping_criteria=0.0001,
                           p_crossover=0.65, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.1, p_point_mutation=0.15,
                           max_samples=0.9, verbose=1,
                           function_set=['add', 'sub', 'mul', 'div',
                                         'neg','log', 'inv', sqrtt, sqr, cub],
                           parsimony_coefficient=0.01, random_state=65432, n_jobs=2)


est_gp.fit(X_train, y_train)
yp=est_gp.predict(X_train)
print(est_gp._program)
print('score', est_gp.score(X_train, y_train)) 

#%%

# Plot the data surface and the trained output
#------------------------------------------------------------------------------

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(60,10))


ax.scatter(X_train[:,0]*10, X_train[:,1], y_train, s=40) 
ax.plot_surface(mx0*10, mx1, yp.reshape(21,7), 
                rstride=1, cstride=1, color='green', alpha=0.5)

plt.title('Real surface vs trained points', fontsize=50, pad=10)


lp = 15
fs = 20

ax.set_zlim(0, 15)
ax.set_ylim(0, 25)

ax.set_zlabel('Vacancies', fontsize=fs, labelpad=lp)
ax.set_xlabel('E', fontsize=fs, labelpad=lp)
ax.set_ylabel('Number of projectile atoms', fontsize=fs, labelpad=lp)

ax.tick_params(labelsize=20)


ax.view_init(20,-120)
plt.show()

#%%

# Plot the image showing pieces of equation
#------------------------------------------------------------------------------

from IPython.display import Image, display
import pydotplus


graph = est_gp._program.export_graphviz()
graph = pydotplus.graphviz.graph_from_dot_data(graph)
display(Image(graph.create_png()))


# Find the equation of the surface
#------------------------------------------------------------------------------

from sympy import *

x, y, z = symbols('x y z', real=True)

locals = {
    'sub':   lambda x, y : x - y,
    'div':   lambda x, y : x/y,
    'mul':   lambda x, y : x*y,
    'add':   lambda x, y : x + y,
    'neg':   lambda x    : -x,
    'pow':   lambda x, y : x**y,
    'inv':   lambda x    : 1/x,
    'sqrtt': lambda x    : np.abs(x)**0.5,
    'sqr':   lambda x    : x**2,
    'cub':   lambda x    : x**3,
    'log':   lambda x    : log(x),
    'exp':   lambda x    : exp(x),
    'X0' :   x/10.0,
    'X1' :   y
}

a = sympify(str(est_gp._program), locals=locals)
print(a)


#%%

# Plot the real surface vs surface made by the equation
#------------------------------------------------------------------------------
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(60,10))

# Make data
X=np.linspace(np.amin(mx0)*10,np.amax(mx0)*10,20)
Y=np.linspace(np.amin(mx1),np.amax(mx1),20)
 
xa,ya = np.meshgrid(X,Y)

z = lambdify([x,y], a, 'numpy')



ax.plot_surface(xa, ya, z(xa,ya), cmap=cm.coolwarm, alpha=0.6, 
                linewidth=0, antialiased=False) 

ax.plot_surface(mx0*10, mx1, yp.reshape(21,7), rstride=1, 
                cstride=1, color='green', alpha=0.8)

plt.title('Real surface vs surface from equation', fontsize=50, pad=10)

lp = 15
fs = 20

ax.set_ylim(0, 25)
ax.set_zlim(0, 15)

ax.set_zlabel('Vacancies', fontsize=fs, labelpad=lp)
ax.set_xlabel('E', fontsize=fs, labelpad=lp)
ax.set_ylabel('Number of projectile atoms', fontsize=fs, labelpad=lp)

ax.tick_params(labelsize=20)

ax.view_init(20,-120)

plt.show()

