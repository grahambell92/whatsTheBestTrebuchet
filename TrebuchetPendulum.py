#To force float division not integer
from __future__ import division
from scipy.integrate import odeint
import numpy as np

from math import pi, sin, cos, sqrt
from matplotlib import animation
from scipy.optimize import minimize
import math
import latexPlotTools as lpt
import matplotlib.pyplot as plt

def ThrowRange(theta,thetadot):

    x2 = -L2*np.sin(theta)
    y2 = -L2*np.cos(theta)

    Vx = L2*thetadot*sin(theta-pi/2)
    Vy = L2*thetadot*cos(theta-pi/2)

    Sx = Vx*((Vy+np.sqrt((Vy**2)+(2*g*(y2-h))))/g)+x2
    if math.isnan(Sx):
        Sx = 0
    #Sx = 2*Vx*Vy/g
    #theta > pi/2 and theta < 3*pi/2 and
    if Vx > 0 and Vy >0 and Sx>0:
        print("Vx: %.3f" % Vx)
        print("Vy: %.3f" % Vy)
        return Sx
    else:
        return 0.0001*Sx

def deriv(z,t):
    I = m1*L1**2 + m2*L2**2
    a = g*(m1*L1 - m2*L2)

    #z[0] = theta
    #z[1] = theta_dot
    dz0 = z[1]
    dz1 = a*sin(z[0])/I
    return np.array([dz0, dz1])

def pendTreb(x):
    global g
    global m1
    global m2
    global L2
    global L1
    global x1,y1,x2,y2,t,Sx,z,h
    g = 9.81
    m1 = x
    m2 = 1
    L2 = 2
    #Set max length of total arm to be 3m.
    L1 = 0.5
    h = -np.cos(45*pi/180)*L2
    endtime = 1
    dt = 300
    t = np.linspace(0,endtime,dt)
    zinit = np.array([45*pi/180, 0])
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
    fig, ax  = lpt.newfig(0.5)
    plt.subplot(1,1,1)
    patches = []
    #subplot(1,1,1)

    #Vertical line
    patches.append(ax.add_line(plt.Line2D((max_index/dt, max_index/dt), (-5,max_value), lw=1,dashes = ([2,2]), color = 'r')))
    #Horizontal line
    patches.append(ax.add_line(plt.Line2D((0, max_index/dt), (max_value,max_value), lw=1,dashes = ([2,2]), color = 'r')))
    plt.plot(t,Sx[:], color="k", marker="o", markersize = 2,linewidth = 0.5)
    ax.set_xlabel(r'Time (s)')
    ax.set_ylabel(r'Range (m)')
    ylimits = plt.ylim(min(Sx)-1,max(Sx)*1.1)
    plt.tight_layout()
    plt.show()
    lpt.savefig('PendulumRange')
    return rangethrown

print(pendTreb(40))
exit(0)

#Plot for Paper:
#fig = plt.figure(figsize=(10,6), dpi=80)
fig, ax  = lpt.newfig(0.5)
plt.subplot(1,1,1,aspect='equal')

xlimits = plt.xlim(-3,18)
ylimits = plt.ylim(h*1.3,5)
ax = plt.axes(xlim = xlimits, ylim = ylimits,aspect='equal')

patches = []
#h stem frame line
patches.append(ax.add_line(plt.Line2D((0, 0), (0,h), lw=1,color = 'k')))
#ground line
patches.append(ax.add_line(plt.Line2D((-4, 18), (h,h), lw=1,color = 'g')))
for i in range(0,200,15):
    #CW circle
    patches.append(ax.add_patch(plt.Circle((x1[i % len(t)],y1[i % len(t)]),0.2,color='k') ))
    #x1 hinge position circle
    patches.append(ax.add_patch(plt.Circle((x2[i % len(t)],y2[i % len(t)]),0.1,color='r') ))


    #L1 line
    patches.append(ax.add_line(plt.Line2D((0, x1[i % len(t)]), (0,y1[i % len(t)]), lw=1,alpha = 0.5)))
    #L2 line
    patches.append(ax.add_line(plt.Line2D((0, x2[i % len(t)]), (0,y2[i % len(t)]), lw=1,alpha = 0.5)))
#Center dot (drawn outside so that it's on top of the lines
patches.append(ax.add_patch(plt.Circle((0,0),0.1,color='k') ))

V0x = L2*z[max_index,1]*np.sin(z[max_index,0]-pi/2)
V0y = L2*z[max_index,1]*np.cos(z[max_index,0]-pi/2)
SyPlot = []
SxPlot = []
tproj = np.linspace(0,((V0y+np.sqrt((V0y**2)+(2*g*(y2[max_index]-h))))/g),100)
#((Vy+np.sqrt((Vy**2)+(2*g*y2)))/g)
for i in range(len(tproj)):
    ithxpos = V0x*tproj[i]+x2[max_index]
    SxPlot.append(ithxpos)
    ithypos = y2[max_index] + V0y*tproj[i] - 0.5*g*tproj[i]**2
    SyPlot.append(ithypos)
plt.plot(SxPlot,SyPlot,dashes = ([2,2]) ,color = 'k',linewidth = 1)
ax.set_xlabel(r'x (m)')
ax.set_ylabel(r'y (m)')
plt.show()
lpt.savefig('PendulumThrowOutline')

'''
#Animate this shit!

fig = plt.figure(figsize=(10,6), dpi=80)
plt.subplot(1,1,1,aspect='equal')

xlimits = plt.xlim(-4,4)
ylimits = plt.ylim(-4,4)
ax = plt.axes(xlim = xlimits, ylim = ylimits,aspect='equal')


def init():

    return []

def animate(i):
    patches = []

    #L1 line
    patches.append(ax.add_line(plt.Line2D((0, x1[i % len(t)]), (0,y1[i % len(t)]), lw=2.5)))
    #L2 line
    patches.append(ax.add_line(plt.Line2D((0, x2[i % len(t)]), (0,y2[i % len(t)]), lw=2.5)))

    #CW circle
    patches.append(ax.add_patch(plt.Circle((x1[i % len(t)],y1[i % len(t)]),0.2,color='k') ))
    #x1 hinge position circle
    patches.append(ax.add_patch(plt.Circle((x2[i % len(t)],y2[i % len(t)]),0.1,color='r') ))

    #Center dot
    patches.append(ax.add_patch(plt.Circle((0,0),0.05,color='k') ))
    #Release dot

    if i > max_index:
        patches.append(ax.add_patch(plt.Circle((-3,3),1,color='r') ))


    return patches

fps = 24
anim = animation.FuncAnimation(fig,animate,init_func=init, frames = len(Sx), interval = 10, blit = True)
plt.show()
'''