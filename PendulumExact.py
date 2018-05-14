__author__ = 'Graham'

from scipy.integrate import odeint, quad
import numpy as np

from math import pi, sin, cos, sqrt, asin
from matplotlib import animation
from matplotlib import pyplot as plt
import scipy
import mpmath
import math

def ThrowRange(theta,thetadot):

    x2 = -L2*np.sin(theta)
    y2 = -L2*np.cos(theta)

    Vx = L2*thetadot*sin(theta-pi/2)
    Vy = L2*thetadot*cos(theta-pi/2)

    Sx = Vx*((Vy+np.sqrt((Vy**2)+(2*g*(y2))))/g)+x2
    if math.isnan(Sx):
        Sx = 0
    #Sx = 2*Vx*Vy/g
    #theta > pi/2 and theta < 3*pi/2 and
    if Vx > 0 and Vy >0 and Sx>0:
        #print "Vx: %.3f" % Vx
        #print "Vy: %.3f" % Vy
        return Sx
    else:
        return 0.0001*Sx

def deriv(z,t):
    #I = m1*L1**2 + m2*L2**2
    #a = g*(m1*L1 - m2*L2)
    I = m1*L1**2
    a = g*(m1*L1)
    #z[0] = theta
    #z[1] = theta_dot
    dz0 = z[1]
    dz1 = -a*sin(z[0])/I
    return np.array([dz0, dz1])

def pendTreb(x):
    global g
    global m1
    global m2
    global L2
    global L1
    global x1,y1,x2,y2,t,Sx,z, endtime, dt, zinit
    g = 9.81
    m1 = x
    m2 = 0
    L2 = 0
    #Set max length of total arm to be 3m.
    L1 = 1
    endtime = 5
    dt = 50
    t = np.linspace(0,endtime,dt)
    zinit = np.array([170*pi/180, 0])
    z = odeint(deriv, zinit, t)

    x1 = L1*np.sin(z[:,0])
    y1 = L1*np.cos(z[:,0])
    x2 = -L2*np.sin(z[:,0])
    y2 = -L2*np.cos(z[:,0])

    Sx = []
    for i in range(0,len(t)):
        Sx.append(ThrowRange(z[i,0],z[i,1]))
        #Grab the largest range value. ThrowRange function args: theta,thetadot,alpha,alphadot
    #Sx = Sx[0:400]
    import operator
    global max_index, max_value
    max_index, max_value = max(enumerate(Sx), key=operator.itemgetter(1))
    rangethrown = max_value
    #figure(figsize=(10,6), dpi=80)
    return rangethrown


print pendTreb(10)

#Exact pendulum

w0 = sqrt(g/L1)
sn = mpmath.ellipfun('sn')
print zinit[0]
theta = []
for i in t:
    #thetaone = 2*asin( sin(zinit[0]/2) *  sn( ((w0/w)*((pi/2)-w*i)) , sin(zinit[0]/2) ) )
    thetaone = 2*asin( sin(zinit[0]/2) *  sn( mpmath.ellipk(sin(zinit[0]/2)**2) -w0*i , sin(zinit[0]/2)**2 ) )
    #thetaone = 2*asin(sin(zinit[0]/2) * sn(w0*i,sin(zinit[0]/2)))
    theta.append(thetaone)

# Linear approximation
w0 = sqrt(g/L1)
thetalinear = []
for i in t:
    thetaone = zinit[0]*cos(w0*i)
    thetalinear.append(thetaone)

print "Period of linear version: "
print 2*pi/w0
plt.figure
plt.plot(t,theta,color = 'k')
plt.plot(t,z[:,0],'r--',linewidth=3.0)
plt.plot(t,thetalinear,color = 'b')
plt.xlabel('t')
plt.ylabel('theta')
plt.show()
