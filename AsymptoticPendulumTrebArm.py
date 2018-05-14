
from scipy.integrate import odeint
import numpy as np
import math
from math import pi, sin, cos, sqrt
from scipy.optimize import minimize
import matplotlib as mpl
from matplotlib import animation

mpl.use('pgf')

def figsize(scale):
    fig_width_pt = 503.169                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)
import matplotlib.pyplot as plt

# I make my own newfig and savefig functions
def newfig(width):
    plt.clf()
    fig = plt.figure(figsize=figsize(width))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename):
    plt.savefig('{}.pgf'.format(filename))
    plt.savefig('{}.pdf'.format(filename))


def ThrowRange(theta,thetadot,L2):

    x2 = -L2*np.sin(theta)
    y2 = -L2*np.cos(theta)

    Vx = L2*thetadot*sin(theta-pi/2)
    Vy = L2*thetadot*cos(theta-pi/2)

    #Sx = Vx*((Vy+np.sqrt((Vy**2)+(2*g*(y2-h))))/g)+x2
    #Sx = 2*Vx*Vy/g
    Sx = thetadot*L2

    if math.isnan(Sx):
        Sx = 0

    #theta > pi/2 and theta < 3*pi/2 and
    if Vx > 0 and Vy >0 and Sx>0:
        #print "Vx: %.3f" % Vx
        #print "Vy: %.3f" % Vy
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
    global x1,y1,x2,y2,t,Sx,h
    g = 9.81
    m1 = 30
    m2 = 1
    L2 = x
    #Set max length of total arm to be 3m.
    L1 = 1
    h = -np.cos(45*pi/180)*L2
    endtime = 7
    dt = 450
    t = np.linspace(0,endtime,dt)
    zinit = np.array([45*pi/180, 0])
    z = odeint(deriv, zinit, t)

    x1 = L1*np.sin(z[:,0])
    y1 = L1*np.cos(z[:,0])
    x2 = -L2*np.sin(z[:,0])
    y2 = -L2*np.cos(z[:,0])




    Sx = []
    for i in range(0,len(t)):
        Sx.append(ThrowRange(z[i,0],z[i,1],L2))
        #Grab the largest range value. ThrowRange function args: theta,thetadot,alpha,alphadot
    #Sx = Sx[0:400]
    rangethrown = max(Sx)
    import operator
    global max_index, max_value
    max_index, max_value = max(enumerate(Sx), key=operator.itemgetter(1))
    rangethrown = max_value
    return rangethrown


ranges = []
size = 50
armLengths = np.linspace(1.5,size,size*2,endpoint = True)
for arms in armLengths:
    ranges.append(pendTreb(arms))
    print arms
ranges = ranges/max(ranges)
fig, ax = newfig(0.5)
plt.subplot(1,1,1)
#plt.xscale('log')
patches = []
#subplot(1,1,1)

#Vertical line
#patches.append(ax.add_line(plt.Line2D((max_index/dt, max_index/dt), (-5,max_value), lw=1,dashes = ([2,2]), color = 'r')))
#Horizontal line at ~100%
#patches.append(ax.add_line(plt.Line2D((0, 5000), (0.5,0.5), lw=0.5,dashes = ([2,2]), color = 'r')))
#patches.append(ax.add_line(plt.Line2D((0, massRatios[-1]), (0.9*max(ranges),0.9*max(ranges)), lw=0.5,dashes = ([2,2]), color = 'r')))
# Yankee Seige 10lbs pumpkin 21,000lbs CW = 2100 m1/m2
#plt.plot([2100,2100],[0,1],color='r',linewidth = 0.5,linestyle = '--')
# American Chucker CW 1400, 10lb pumpkin = 140
#plt.plot([140,140],[0,1],color='r',linewidth = 0.5,linestyle = '--')
plt.plot(armLengths,ranges, color="k", marker="o", markersize = 2,markerfacecolor='w', linestyle = "None")
ax.set_xlabel(r'L2 Arm')
ax.set_ylabel(r'Range/ Maximum range')
ylimits = plt.ylim(0.1,1.1)
xlimits = plt.xlim(0,size)
print "Max Range: %.3f" % max(ranges)
indexmax = ranges.argmax()
print "Index %d" %indexmax
print "Arm for max range: %.3f" % armLengths[indexmax]

'''
annotate(r'',
         xy=(200, max(ranges)), xycoords='data',
         xytext=(+10, +30), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
'''

plt.tight_layout()
plt.show()
savefig('AsymptoticArmRatioTreb')

'''
# Max Throw range mass ratio:
import operator
max_index, max_value = max(enumerate(ranges), key=operator.itemgetter(1))
# max index is the index of the maximum throw range.

maxThrowMassRatio = massRatios[max_index]
fullThrowRange = ranges[max_index]
percent90ThrowRange = 0.9*fullThrowRange
print "Max Throw mass ratio:"
print maxThrowMassRatio
print "Full throw range:"
print fullThrowRange
print "90% of full throw range:"
print percent90ThrowRange
print "Treb throw at theoretical throw range of m1/m2 = 56:"
print pendTreb(56)
# Is the ratio at 90% of max range equal to the mass ratio predicted?

'''

   #Animate this shit!


fig = plt.figure(figsize=(10,6), dpi=80)
plt.subplot(1,1,1,aspect='equal')

xlimits = plt.xlim(-4,4)
ylimits = plt.ylim(-4,4)
ax = plt.axes(xlim = xlimits, ylim = ylimits,aspect='equal')

'''
def init():

    return []

def animate(i):
    patches = []

    #L1 line
    patches.append(ax.add_line(plt.Line2D((0, x1[i % len(t)]), (0,y1[i % len(t)]), lw=2.5)))
    #L2 line
    patches.append(ax.add_line(plt.Line2D((0, x2[i % len(t)]), (0,y2[i % len(t)]), lw=2.5)))

    #x1 hinge position circle
    patches.append(ax.add_patch(plt.Circle((x1[i % len(t)],y1[i % len(t)]),0.1,color='b') ))
    #CW circle
    patches.append(ax.add_patch(plt.Circle((x2[i % len(t)],y2[i % len(t)]),0.1,color='g') ))

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