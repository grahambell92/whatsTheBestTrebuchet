__author__ = 'Graham'

# Constant radius Murlin type trebuchet
# Precursor to the variable Murlin since I'm struggling to get the equations right.
# 09/11/2015

from scipy.integrate import odeint
import numpy as np
import pylab
from math import pi, sin, cos, sqrt
from matplotlib import animation
from matplotlib import pyplot as plt

def deriv(z,t):
    #z[0] = theta
    #z[1] = alpha
    #z[2] = theta_dot
    #z[3] = alpha_dot

    dz0 = z[2]
    dz1 = z[3]
    dz2 = 3*(-2*L1*g*m1 + L2**2*m3*sin(2*z[1] - 2*z[0])*z[2]**2 + 2*L2*L3*m3*cos(z[1] - z[0])*z[3]**2 + L2*g*m3*sin(2*z[1] - z[0]) + L2*g*m3*sin(z[0]) + L2*g*m_b*sin(z[0]))/(2*(-3*L1**2*m1 + 3*L2**2*m3*sin(z[1] - z[0])**2 - 3*L2**2*m3 - L2**2*m_b))
    dz3 = (-3*L2*(-2*L1*g*m1 + 2*L2*L3*m3*cos(z[1] - z[0])*z[3]**2 + 2*L2*g*m3*sin(z[0]) + L2*g*m_b*sin(z[0]))*sin(z[1] - z[0]) + (L2*cos(z[1] - z[0])*z[2]**2 + g*cos(z[1]))*(-6*L1**2*m1 - 6*L2**2*m3 - 2*L2**2*m_b))/(2*L3*(-3*L1**2*m1 + 3*L2**2*m3*sin(z[1] - z[0])**2 - 3*L2**2*m3 - L2**2*m_b))
    return np.array([dz0,dz1,dz2,dz3])

g = 9.81
m1 = 1000
m3 = 1

L1 = 0.3
x4 = 1
L2 = 1
L3 = 1
beamDensity = 1
m_b = beamDensity*(L3)

endtime = 5
dt = 0.01
t = np.linspace(0,endtime,endtime/dt)
zinit = np.array([45*pi/180, 1*pi/180, 0, 0])
z = odeint(deriv, zinit, t)

xb = -L2/2*np.sin(z[:,0])
yb = -L2/2*np.cos(z[:,0])

x2 = -L2*np.sin(z[:,0])
y2 = -L2*np.cos(z[:,0])

x3 = x2 + L3*np.cos(z[:,1])
y3 = y2 - L3*np.sin(z[:,1])

#L4 = L1 * z[:,0] + np.sqrt(x4**2 - L1**2) #- L1 * zinit[0]
y4 = -L1*z[:,0]
#y4 = -np.sqrt(L4**2 + L1**2 - x4**2 + 0.001)
#Attach Position for the string
#beta0 = np.arccos(L1/x4)
#delta = np.arctan2(y4,x4)
#xL1 = L1*np.cos(beta0 - delta)
#yL1 = L1*np.sin(beta0 - delta)

#Animate this shit!

fig = plt.figure(figsize=(10,6), dpi=80)
plt.subplot(1,1,1,aspect='equal')

xlimits = plt.xlim(-4,4)
ylimits = plt.ylim(-4,4)
ax = plt.axes(xlim = xlimits, ylim = ylimits,aspect='equal')

print(z)
def init():

    return []

def animate(i):
    patches = []

    #Vertical and horizontal lines:
    #Horizontal rail
    patches.append(ax.add_line(plt.Line2D((-1, 1), (0, 0), lw=1.7, alpha = 0.3, color='k',linestyle = '--')))
    #Vertical Rail
    patches.append(ax.add_line(plt.Line2D((0, 0), (-1.5, 1.5), lw=1.7, alpha = 0.3, color='k',linestyle = '--')))

    #0, 0 - x2,y2 line
    patches.append(ax.add_line(plt.Line2D((0, x2[i % len(t)]), (0,y2[i % len(t)]), lw=2.5)))

    #x2,y2-x3,y3 line
    patches.append(ax.add_line(plt.Line2D((x2[i % len(t)], x3[i % len(t)]), (y2[i % len(t)], y3[i % len(t)]), lw=2.5)))

    #Draw cable to mass
    #patches.append(ax.add_line(plt.Line2D((xL1[i % len(t)], x4), (yL1[i % len(t)], y4[i % len(t)]), lw=1.8)))

    #m1 CW circle
    patches.append(ax.add_patch(plt.Circle((x4,y4[i % len(t)]),0.2,color='k')))
    #patches.append(ax.add_patch(plt.Circle((1,-1),0.2,color='k')))
    #Centre Cam Circle
    patches.append(ax.add_patch(plt.Circle((0,0),L1,color='r') ))

    #beam centre circle
    patches.append(ax.add_patch(plt.Circle((xb[i % len(t)],yb[i % len(t)]),0.05,color='k') ))

    #Projectile circle
    patches.append(ax.add_patch(plt.Circle((x3[i % len(t)],y3[i % len(t)]),0.1,color='r') ))

    return patches

fps = 24
anim = animation.FuncAnimation(fig,animate,init_func=init, frames = len(t), interval = 10, blit = True)
plt.show()