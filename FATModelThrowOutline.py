__author__ = 'Graham'

# Full Treb Modelling.
# Equations derived symbolically through sympy.

import numpy as np
from math import pi, sin, cos
import latexPlotTools as lpt
import matplotlib.pyplot as plt
from FATTrebClass import FATTreb

FAT = FATTreb("treb1")
zinit = np.array([45.0 * pi / 180, 1.0 * pi / 180, 0.0, 0.0])
L2 = 1.8
L3 = 1.8
g = 9.81
rangeThrown = FAT.runFATTreb(m1=100, L2 = L2, L3 = L3, zinit=zinit, endtime = 0.8)
h, t = FAT.h, FAT.t
y1, y2, y3 = FAT.y1, FAT.y2, FAT.y3
x2, x3, x4 = FAT.x2, FAT.x3, FAT.x4
xb, yb = FAT.xb, FAT.yb
z = FAT.z
max_index = FAT.max_index

#Plot Throw Outline for paper
fig, ax  = lpt.newfig(0.6)
plt.subplot(1,1,1,aspect='equal')

xlimits = plt.xlim(-5,5)
ylimits = plt.ylim(-2,5)
ax = plt.axes(xlim = xlimits, ylim = ylimits,aspect='equal')

patches = []

#Vertical and horizontal lines:
#Horizontal rail
patches.append(ax.add_line(plt.Line2D((-5, 5), (0, 0), lw=0.7, alpha = 0.5, color='k',dashes = ([2,2]))))
#Vertical Rail
patches.append(ax.add_line(plt.Line2D((0, 0), (-5, 5), lw=0.7, alpha = 0.5, color='k',dashes = ([2,2]))))
#Ground
patches.append(ax.add_line(plt.Line2D((-20, 20), (h, h), lw=0.7, color='g')))
for i in np.arange(0,len(t),3):
    #linewidths
    trebarmwidth = 1
    #x4,y4 - x1,y1 line
    patches.append(ax.add_line(plt.Line2D((x4[i % len(t)], 0), (0,y1[i % len(t)]), lw=trebarmwidth, alpha = 0.5)))
    #x4,y4-x2,y2 line
    patches.append(ax.add_line(plt.Line2D((x2[i % len(t)], x4[i % len(t)]), (y2[i % len(t)], 0), lw=trebarmwidth, alpha = 0.5)))
    #x2,y2-x3,y3 line
    patches.append(ax.add_line(plt.Line2D((x2[i % len(t)], x3[i % len(t)]), (y2[i % len(t)], y3[i % len(t)]), lw=trebarmwidth, alpha = 0.5)))

    #m1 CW circle
    patches.append(ax.add_patch(plt.Circle((0,y1[i % len(t)]),0.4,color='k', alpha = 0.2)))
    #Beam axle circle
    patches.append(ax.add_patch(plt.Circle((x4[i % len(t)],0),0.08,color='r', alpha = 0.3) ))
    #beam centre circle
    patches.append(ax.add_patch(plt.Circle((xb[i % len(t)],yb[i % len(t)]),0.05,color='k', alpha = 0.5) ))
    #Projectile circle
    patches.append(ax.add_patch(plt.Circle((x3[i % len(t)],y3[i % len(t)]),0.15,color='r') ))
    #Release dot

# Plot the proj trajectory curve.
#Maximum index calculating the initial V0x and V0y
# gamma = 3*pi/2 - alpha
    #z[0] = theta
    #z[1] = alpha
    #z[2] = theta_dot
    #z[3] = alpha_dot

V0x = L2*z[max_index,2]*np.cos(np.pi - z[max_index,0]) + L3*z[max_index,3]*np.cos(3*np.pi/2 - z[max_index,1])
V0y = L2*z[max_index,2]*np.sin(np.pi - z[max_index,0]) + L3*z[max_index,3]*np.sin(3*np.pi/2 - z[max_index,1])
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
ax.set_xlabel(r'$x$ (m)')
ax.set_ylabel(r'$y$ (m)')
plt.tight_layout()
lpt.savefig('FATThrowOutline')
