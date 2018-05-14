__author__ = 'Graham'

# Full Treb Modelling.
# Equations derived symbolically through sympy.

from scipy.integrate import odeint
import numpy as np
from math import pi, sin, cos
import math
import latexPlotTools as lpt
import matplotlib.pyplot as plt

def ThrowRange(theta,thetadot,alpha,alphadot):

    x2 = -L2*np.sin(theta)
    y2 = -L2*np.cos(theta)
    x3 = x2+L3*np.cos(alpha)
    y3 = y2 - L3*np.sin(alpha)

    gamma = 3*pi/2 - alpha
    Vx = L2*thetadot*cos(pi - theta) + L3*alphadot*cos(gamma)
    Vy = L2*thetadot*sin(pi - theta) + L3*alphadot*sin(gamma)

    Sx = Vx*((Vy+np.sqrt((Vy**2)+(2*g*(y3-h))))/g)+x3
    if math.isnan(Sx):
        Sx = 0
    #theta > pi/2 and theta < 3*pi/2 and
    if Vx > 0 and Vy >0 and Sx>0 and alpha < 2*pi and y3 > 0:# and thetadot > 0:
        #print "Vx: %.3f" % Vx
        #print "Vy: %.3f" % Vy
        return Sx
    else:
        return 0.00000001*Sx

def deriv(z,t):
    #z[0] = theta
    #z[1] = phi
    #z[2] = alpha
    #z[3] = theta_dot
    #z[4] = phi_dot
    #z[5] = alpha_dot

    dz0 = z[3]
    dz1 = z[4]
    dz2 = z[5]
    dz3 = 3*(L1**2*m1*sin(2*z[1] + 2*z[0])*z[3]**2 - 2*L1*L4*m1*sin(z[1] + z[0])*z[4]**2 - L1*g*m1*sin(2*z[1] + z[0]) - L1*g*m1*sin(z[0]) - L1*g*m_b*sin(z[0]) + L2**2*m3*sin(2*z[2] - 2*z[0])*z[3]**2 + 2*L2*L3*m3*cos(z[2] - z[0])*z[5]**2 + L2*g*m3*sin(2*z[2] - z[0]) + L2*g*m3*sin(z[0]) + L2*g*m_b*sin(z[0]))/(2*(3*L1**2*m1*cos(z[1] + z[0])**2 - 3*L1**2*m1 - L1**2*m_b + L1*L2*m_b + 3*L2**2*m3*sin(z[2] - z[0])**2 - 3*L2**2*m3 - L2**2*m_b))
    dz4 = (-3*L1*L2*m3*(L2*cos(z[2] - z[0])*z[3]**2 + g*cos(z[2]))*sin(z[2] - z[0])*cos(z[1] + z[0]) + 3*L1*(2*L1*L4*m1*sin(z[1] + z[0])*z[4]**2 + 2*L1*g*m1*sin(z[0]) + L1*g*m_b*sin(z[0]) - 2*L2*L3*m3*cos(z[2] - z[0])*z[5]**2 - 2*L2*g*m3*sin(z[0]) - L2*g*m_b*sin(z[0]))*cos(z[1] + z[0])/2 + (L1*sin(z[1] + z[0])*z[3]**2 - g*sin(z[1]))*(-3*L1**2*m1 - L1**2*m_b + L1*L2*m_b + 3*L2**2*m3*sin(z[2] - z[0])**2 - 3*L2**2*m3 - L2**2*m_b))/(L4*(3*L1**2*m1*cos(z[1] + z[0])**2 - 3*L1**2*m1 - L1**2*m_b + L1*L2*m_b + 3*L2**2*m3*sin(z[2] - z[0])**2 - 3*L2**2*m3 - L2**2*m_b))
    dz5 = (-3*L1*L2*m1*(L1*sin(z[1] + z[0])*z[3]**2 - g*sin(z[1]))*sin(z[2] - z[0])*cos(z[1] + z[0]) + 3*L2*(2*L1*L4*m1*sin(z[1] + z[0])*z[4]**2 + 2*L1*g*m1*sin(z[0]) + L1*g*m_b*sin(z[0]) - 2*L2*L3*m3*cos(z[2] - z[0])*z[5]**2 - 2*L2*g*m3*sin(z[0]) - L2*g*m_b*sin(z[0]))*sin(z[2] - z[0])/2 + (L2*cos(z[2] - z[0])*z[3]**2 + g*cos(z[2]))*(3*L1**2*m1*cos(z[1] + z[0])**2 - 3*L1**2*m1 - L1**2*m_b + L1*L2*m_b - 3*L2**2*m3 - L2**2*m_b))/(L3*(3*L1**2*m1*cos(z[1] + z[0])**2 - 3*L1**2*m1 - L1**2*m_b + L1*L2*m_b + 3*L2**2*m3*sin(z[2] - z[0])**2 - 3*L2**2*m3 - L2**2*m_b))
    return np.array([dz0,dz1,dz2,dz3,dz4,dz5])

def FullSlingTreb(x):
    global g
    global L2
    global L3
    global m3
    global L1
    global L4
    global m_b,m1
    global endtime
    global dt
    global t,Sx, z
    global x1,y1,x2,y2,x3,y3,x4,y4,h

    g = 9.81
    m1 = x
    m3 = 1
    L1 = 1
    L2 = 2
    L3 = 2

    beamDensity = 1
    m_b = (L1 + L2)* beamDensity
    h = -np.cos(45*pi/180)*L2
    L4 = - L1 - h
    endtime = 1.5
    dt = 250
    t = np.linspace(0,endtime,dt)
    zinit = np.array([45*pi/180, 1*pi/180, 1*pi/180,0, 0, 0])
    z = odeint(deriv, zinit, t)

    x1 = L1*np.sin(z[:,0])
    y1 = L1*np.cos(z[:,0])
    x2 = -L2*np.sin(z[:,0])
    y2 = -L2*np.cos(z[:,0])
    x3 = x2+L3*np.cos(z[:,2])
    y3 = y2 - L3*np.sin(z[:,2])

    x4 = x1+L4*np.sin(z[:,1])
    y4 = y1-L4*np.cos(z[:,1])
    yb = -((L2-L1)/2)*np.cos(z[:,0])

    rangethrown = 0
    Sx = []

    for i in range(0,len(t)):
        Sx.append(ThrowRange(z[i,0],z[i,3],z[i,2],z[i,5]))
        #Grab the largest range value. ThrowRange function args: theta,thetadot,alpha,alphadot

    rangethrown = max(Sx)
    import operator
    global max_index, max_value
    max_index, max_value = max(enumerate(Sx), key=operator.itemgetter(1))

    rangethrown = max_value
    #print "L2: %.3f L3: %.3f L4 %.3f" % (L2, L3, L4)
    #print "Range: %.3f" % rangethrown
    #print "max_index: %.d" % max_index

    #Minimization algorithm. Set negative value.
    return rangethrown

print(FullSlingTreb(100))

#Plot Throw Outline for paper
#fig = plt.figure(figsize=(10,6), dpi=80)
fig, ax  = lpt.newfig(0.6)
plt.subplot(1,1,1,aspect='equal')

xlimits = plt.xlim(-6,6)
ylimits = plt.ylim(h*1.3,6)
ax = plt.axes(xlim = xlimits, ylim = ylimits,aspect='equal')

patches = []

#Vertical and horizontal lines:
#Horizontal rail
#patches.append(ax.add_line(plt.Line2D((-5, 5), (0, 0), lw=0.7, alpha = 0.5, color='k',dashes = ([2,2]))))
#Vertical Rail
#patches.append(ax.add_line(plt.Line2D((0, 0), (-5, 5), lw=0.7, alpha = 0.5, color='k',dashes = ([2,2]))))
#Ground
patches.append(ax.add_line(plt.Line2D((-20, 20), (h, h), lw=0.7, color='g')))
for i in range(0,150,11):
    #linewidths
    trebarmwidth = 1
    armalpha = 0.5
    #L1 line
    patches.append(ax.add_line(plt.Line2D((0, x1[i % len(t)]), (0,y1[i % len(t)]), lw=trebarmwidth,alpha = armalpha)))
    #L2 line
    patches.append(ax.add_line(plt.Line2D((0, x2[i % len(t)]), (0,y2[i % len(t)]), lw=trebarmwidth)))
    #L4 line
    patches.append(ax.add_line(plt.Line2D((x1[i % len(t)], x4[i % len(t)]), (y1[i % len(t)],y4[i % len(t)]), lw=trebarmwidth,alpha = armalpha)))
    #L3 line
    patches.append(ax.add_line(plt.Line2D((x2[i % len(t)], x3[i % len(t)]), (y2[i % len(t)],y3[i % len(t)]), lw=trebarmwidth,color='c',alpha = armalpha)))

    #x1 hinge position circle
    #patches.append(ax.add_patch(plt.Circle((x1[i % len(t)],y1[i % len(t)]),0.1,color='b') ))
    #CW circle
    patches.append(ax.add_patch(plt.Circle((x4[i % len(t)],y4[i % len(t)]),0.2,color='k') ))
    #Beam tip circle
    #patches.append(ax.add_patch(plt.Circle((x2[i % len(t)],y2[i % len(t)]),0.1,color='r') ))
    #Projectile tip circle
    patches.append(ax.add_patch(plt.Circle((x3[i % len(t)],y3[i % len(t)]),0.1,color='r') ))
    #Center dot
    patches.append(ax.add_patch(plt.Circle((0,0),0.05,color='k') ))




# Plot the proj trajectory curve.
#Maximum index calculating the initial V0x and V0y
# gamma = 3*pi/2 - alpha
    #z[0] = theta
    #z[1] = phi
    #z[2] = alpha
    #z[3] = theta_dot
    #z[4] = phi_dot
    #z[5] = alpha_dot
V0x = L2*z[max_index,3]*np.cos(np.pi - z[max_index,0]) + L3*z[max_index,5]*np.cos(3*np.pi/2 - z[max_index,2])
V0y = L2*z[max_index,3]*np.sin(np.pi - z[max_index,0]) + L3*z[max_index,5]*np.sin(3*np.pi/2 - z[max_index,2])
print("V0x: %.3f V0y: %.3f" % (V0x,V0y))
SyPlot = []
SxPlot = []
tproj = np.linspace(0,((V0y+np.sqrt((V0y**2)+(2*g*(y3[max_index]-h))))/g),100)
#((Vy+np.sqrt((Vy**2)+(2*g*y2)))/g)
for i in range(len(tproj)):
    ithxpos = V0x*tproj[i]+x3[max_index]
    SxPlot.append(ithxpos)
    ithypos = y3[max_index] + V0y*tproj[i] - 0.5*g*tproj[i]**2
    SyPlot.append(ithypos)
plt.plot(SxPlot,SyPlot,dashes = ([2,2]) ,color = 'k',linewidth = 1)
ax.set_xlabel(r'x (m)')
ax.set_ylabel(r'y (m)')
plt.tight_layout()
lpt.savefig('StandardTrebThrowOutline')





