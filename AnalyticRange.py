
import mpmath
import matplotlib as mpl
import numpy as np
from math import pi, sin, cos, sqrt, asin
import math
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


# This file will plot the range contour for us to visualize the properties.
theta0 = 45*pi/180
z = sin(theta0/2)
g = 9.81
size = 50
Tspace = np.linspace(0,5,size)
L2lengths = np.linspace(0.1,8,size)
rangeSpace = np.zeros((size,size))
L2Count = 0
tCount = 0
m2 = 1
m1 = 200
L1 = 1
for t in Tspace:
    for L2 in L2lengths:
        w0 = sqrt((g*(m1*L1-m2*L2))/(m1*L1**2 + m2*L2**2))
        theta = 2*asin(z*mpmath.ellipfun('sn', w0*t, z))
        theta_dot = 2*z*w0*mpmath.ellipfun('cn', w0*t, z)
        range = (2*L2**2*(theta_dot)**2*sin(theta-pi/2)*cos(theta-pi/2))/g
        rangeSpace[tCount,L2Count] = range
        L2Count += 1
    L2Count = 0
    tCount += 1

X, Y = np.meshgrid(Tspace, L2lengths)

fig, ax  = newfig(0.5)
plt.subplot(1,1,1)
i,j = np.unravel_index(rangeSpace.argmax(), rangeSpace.shape)
levels = np.linspace(rangeSpace.min(),rangeSpace.max()+0.01,size*2,endpoint = True)
patches = []
#subplot(1,1,1)
colormap = plt.cm.get_cmap('jet', len(levels)-1)
# Plot 3 times reduces the white lines around each contour level

CS = plt.contourf(X, Y, rangeSpace, levels, cmap=colormap, linestyle = 'None')
CS = plt.contourf(X, Y, rangeSpace, levels, cmap=colormap, linestyle = 'None')
CS = plt.contourf(X, Y, rangeSpace, levels, cmap=colormap, linestyle = 'None')
plt.colorbar(ticks = np.linspace(rangeSpace.min(),rangeSpace.max(),6,endpoint = True))
ax.set_xlabel(r'Time')
ax.set_ylabel(r'L2')
patches = []
print "Max Value .max: %.3f" % rangeSpace.max()
print "Max Value unravel: %.3f" % rangeSpace[i,j]
#print "Max i, j %.3f %.3f" % (L2Space[i],L3Space[j])

patches.append(ax.add_patch(plt.Circle((Tspace[i],L2lengths[j]),0.05,color='y')))

plt.tight_layout()
savefig('AnalyticRange')