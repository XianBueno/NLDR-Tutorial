"""
Sample points from various manifolds and spaces
Created on Sun Apr 23 17:08:21 2017 @author: Xian
"""

import numpy as np
from numpy import pi
from numpy import linalg as LA
from numpy import random as rand

# N will be the number of sample points unless otherwise specified

def arclength(path, onlyTotal=False):
    # Approximates the arclength parametrization emprically by segments
    # `path' should have columns be coordinates, rows observation
    N = path.shape[0]
    path = path.reshape((N,-1))
    s = np.zeros(N)
    for i in range(N-1):
        s[i+1] = s[i] + LA.norm( path[i+1,:] - path[i,:] )
    if onlyTotal:
        return s[-1]    # Final value is total arclength
    else:
        return s

def line(N=1000, a=0, b=1):
    # Sample points on the interval [a,b] in R
    return np.linspace(a,b,num=N).reshape(-1,1)
    # Not tested

def circle(N=1000, minangle=0, maxangle=2*pi, r=1, cx=0, cy=0, get_t=True):
    # Sample points on a circle of radius=r and center=(cx,cy)
    # minangle and maxangle should be in radians
    t = np.linspace(minangle,maxangle,num=N)
    x = r*np.cos(t)+cx; y = r*np.sin(t)+cy
    if get_t:
        return np.column_stack((t,x,y))
    else:
        return np.column_stack((x,y))

def hopf_link(N=1000):
    # ADD    
    return 0
    

def spiral(N=1000, w=1, noise=0,s=0, ax=1, ay=1, minangle=0, maxangle=2*pi, byArclength=False):
    # Sample points from a spiral
    # Wraps around w many times (for default angle range)
    # Parameter 0<=s<1 shifts start point to x=s  (for default angle range)
    # Parameters ax and ay control vertical/horizontal stretch
    # WARNING: This sample is not uniform
    # WARNING: If ax<0 or ay<0 may self-intersect
    # noise makes spiral fuzzier with guassian noise
    if s<0 or 1<= s:
        print('Require 0<= s < 1')
        return 
    t = np.linspace(minangle,maxangle,num=N)
    u = 2*pi*s/(1-s) 
    x = ax*(t+u)/(2*pi+u)*np.cos(w*t) + noise*np.random.normal(0,1,N)
    y = ay*(t+u)/(2*pi+u)*np.sin(w*t) + noise*np.random.normal(0,1,N)
    path = np.column_stack((x,y))
    if byArclength and noise==0:
        s = arclength(path)
        return np.column_stack( (s,path) )
    else:
        return np.column_stack((t,path))

def neilparabola(N=500,length = 1, a=1, byArclength=True):
    # Neil's Parabola is a curve given by y=(2/3)ax^{3/2}
    # and is interesting because we can explicitly 
    # parametrize by arclength
    if byArclength:
        s = np.linspace(0,length,N)
        x = (1/a**2)*(1.5*s*a**2+1)**(2/3)-1/a**2
        y = (2/(3*a**2))*( (1.5*s*a**2+1)**(2/3)-1 )**(3/2)
        return np.column_stack((s,x,y))
    else:
        tmax = (1/a**2)*(1.5*length*a**2+1)**(2/3)-1/a**2
        t = np.linspace(0,tmax,N)
        x = t
        y = (2/3)*a*t**(3/2)
        return np.column_stack((t,x,y))
    
    
def trefoil(N=500, noise=0, tsampletype='even', tmin=0, tmax=2*pi,
            includeLast=False, mu=pi, sigma=1, method='nice'):
    # Samples trefoil knot various ways
    # The 'uniform' and 'guassian' settings are only uniform
    # and guassian on the interval (not necessarily on knot)
    #   'includeLast' option is only for 'evenly',
    #   N evenly spaced points either way but
    #    if includeLast=False: t<tmax; else: t<=tmax
    
    if tsampletype == 'even':
        #t  = np.linspace(0,2*pi,num=N+1)[:-1]
        if includeLast:
            t  = np.linspace(tmin,tmax,num=N)
        else:
            t  = np.linspace(tmin,tmax,num=N+1)[:-1]
    elif tsampletype == 'uniform':
        t  = rand.uniform(tmin,tmax,N)
    elif tsampletype == 'gaussian':
        t  = rand.normal(mu,sigma,N)
    elif tsampletype == 'gaussian_clamp':
        t  = np.clip( rand.normal(mu,sigma,N), tmin, tmax )
    else:
        print('sampletype seems to have wrong value')
    
    # Make trefoil    
    if method == 'nice':
        x = np.sin(t) + 2*np.sin(2*t) + noise*np.random.normal(0,1,N)
        y = np.cos(t) - 2*np.cos(2*t) + noise*np.random.normal(0,1,N)
        z = -np.sin(3*t) + noise*np.random.normal(0,1,N)
    elif method == 'torus':
        x = (2+np.cos(3*t))*np.cos(2*t) + noise*np.random.normal(0,1,N)
        y = (2+np.cos(3*t))*np.sin(2*t) + noise*np.random.normal(0,1,N)
        z = np.sin(3*t) + noise*np.random.normal(0,1,N)
        
    return np.column_stack((t,x,y,z))

#def spunTrefoil(N=500):

def flatTorus(N=1600, r1=1, r2=1):
    # Sample ~N points from a flat torus (in R^4) with radii r1 & r2
    # not uniform in case of r1 != r2
    Ngrid = int(np.sqrt(N))
    grid  = np.linspace(0,2*np.pi,num=Ngrid)
    t = np.array([[x,y] for x in grid for y in grid])
    x = r1*np.cos(t[:,0]); y = r1*np.sin(t[:,0])
    z = r2*np.cos(t[:,1]); w = r2*np.sin(t[:,1]) 
    return np.column_stack((x,y,z,w))    
