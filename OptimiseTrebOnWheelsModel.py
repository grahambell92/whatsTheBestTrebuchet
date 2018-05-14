__author__ = 'Graham'

# Full Treb Modelling.
# Equations derived symbolically through sympy.

from scipy.integrate import odeint
import numpy as np
from pylab import *
from math import pi, sin, cos, sqrt
from matplotlib import animation
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from matplotlib.legend_handler import HandlerLine2D

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
    plt.savefig('{}.pgf'.format(filename), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig('{}.pdf'.format(filename), bbox_extra_artists=(lgd,), bbox_inches='tight')

def ThrowRange(theta,thetadot,alpha,alphadot,x0_dot):

    x2 = -L2*np.sin(theta)
    y2 = -L2*np.cos(theta)
    x3 = x2+L3*np.cos(alpha)
    y3 = y2 - L3*np.sin(alpha)

    gamma = 3*pi/2 - alpha
    Vx = L2*thetadot*cos(pi - theta) + L3*alphadot*cos(gamma) +x0_dot
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
    #z[1] = alpha
    #z[2] = phi
    #z[3] = x0
    #z[4] = theta_dot
    #z[5] = alpha_dot
    #z[6] = phi_dot
    #z[7] = x0_dot

    dz0 = z[4]
    dz1 = z[5]
    dz2 = z[6]
    dz3 = z[7]
    dz4 = 3*(-2*m1*(L1*sin(z[2] + z[0])*z[4]**2 - g*sin(z[2]))*(L1*m3*sin(z[1])**2*cos(z[2] + z[0]) - L1*(m1 + m3 + m_b)*cos(z[2] + z[0]) + L2*m3*sin(z[1] - z[0])*sin(z[1])*cos(z[2]) + (L1*m1 - L2*m3 - Lb*m_b)*cos(z[2])*cos(z[0])) - 2*m3*(L2*cos(z[1] - z[0])*z[4]**2 + g*cos(z[1]))*(L1*m1*sin(z[1])*cos(z[2] + z[0])*cos(z[2]) + L2*m1*sin(z[1] - z[0])*cos(z[2])**2 - L2*(m1 + m3 + m_b)*sin(z[1] - z[0]) + (-L1*m1 + L2*m3 + Lb*m_b)*sin(z[1])*cos(z[0])) - 2*(L1*m1*cos(z[2] + z[0])*cos(z[2]) - L2*m3*sin(z[1] - z[0])*sin(z[1]) + (-L1*m1 + L2*m3 + Lb*m_b)*cos(z[0]))*(L1*m1*sin(z[0])*z[4]**2 - L2*m3*sin(z[0])*z[4]**2 + L3*m3*cos(z[1])*z[5]**2 + L4*m1*sin(z[2])*z[6]**2 - Lb*m_b*sin(z[0])*z[4]**2) + (-m1*sin(z[2])**2 + m3*sin(z[1])**2 - m3 - m_b)*(2*L1*L4*m1*sin(z[2] + z[0])*z[6]**2 + 2*L1*g*m1*sin(z[0]) - 2*L2*L3*m3*cos(z[1] - z[0])*z[5]**2 - 2*L2*g*m3*sin(z[0]) + Lb**2*m_b*sin(2*z[0])*z[4]**2 - 2*Lb*g*m_b*sin(z[0])))/(2*(-3*L1**2*m1*m3*sin(z[1])**2*cos(z[2] + z[0])**2 + 3*L1**2*m1*(m1 + m3 + m_b)*cos(z[2] + z[0])**2 - 6*L1*L2*m1*m3*sin(z[1] - z[0])*sin(z[1])*cos(z[2] + z[0])*cos(z[2]) + 6*L1*m1*(-L1*m1 + L2*m3 + Lb*m_b)*cos(z[2] + z[0])*cos(z[2])*cos(z[0]) - 3*L2**2*m1*m3*sin(z[1] - z[0])**2*cos(z[2])**2 + 3*L2**2*m3*(m1 + m3 + m_b)*sin(z[1] - z[0])**2 - 6*L2*m3*(-L1*m1 + L2*m3 + Lb*m_b)*sin(z[1] - z[0])*sin(z[1])*cos(z[0]) + m1*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*cos(z[2])**2 + m3*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*sin(z[1])**2 - (m1 + m3 + m_b)*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2) + 3*(-L1*m1 + L2*m3 + Lb*m_b)**2*cos(z[0])**2))
    dz5 = (-2*m1*(L1*sin(z[2] + z[0])*z[4]**2 - g*sin(z[2]))*(3*L1*L2*(m1 + m3 + m_b)*sin(z[1] - z[0])*cos(z[2] + z[0]) - 3*L1*(-L1*m1 + L2*m3 + Lb*m_b)*sin(z[1])*cos(z[2] + z[0])*cos(z[0]) + 3*L2*(-L1*m1 + L2*m3 + Lb*m_b)*sin(z[1] - z[0])*cos(z[2])*cos(z[0]) - (3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*sin(z[1])*cos(z[2])) + 2*(L2*cos(z[1] - z[0])*z[4]**2 + g*cos(z[1]))*(3*L1**2*m1*(m1 + m3 + m_b)*cos(z[2] + z[0])**2 + 6*L1*m1*(-L1*m1 + L2*m3 + Lb*m_b)*cos(z[2] + z[0])*cos(z[2])*cos(z[0]) + m1*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*cos(z[2])**2 - (m1 + m3 + m_b)*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2) + 3*(-L1*m1 + L2*m3 + Lb*m_b)**2*cos(z[0])**2) + 2*(3*L1**2*m1*sin(z[1])*cos(z[2] + z[0])**2 + 3*L1*L2*m1*sin(z[1] - z[0])*cos(z[2] + z[0])*cos(z[2]) + 3*L2*(-L1*m1 + L2*m3 + Lb*m_b)*sin(z[1] - z[0])*cos(z[0]) - (3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*sin(z[1]))*(L1*m1*sin(z[0])*z[4]**2 - L2*m3*sin(z[0])*z[4]**2 + L3*m3*cos(z[1])*z[5]**2 + L4*m1*sin(z[2])*z[6]**2 - Lb*m_b*sin(z[0])*z[4]**2) - 3*(L1*m1*sin(z[1])*cos(z[2] + z[0])*cos(z[2]) + L2*m1*sin(z[1] - z[0])*cos(z[2])**2 - L2*(m1 + m3 + m_b)*sin(z[1] - z[0]) + (-L1*m1 + L2*m3 + Lb*m_b)*sin(z[1])*cos(z[0]))*(2*L1*L4*m1*sin(z[2] + z[0])*z[6]**2 + 2*L1*g*m1*sin(z[0]) - 2*L2*L3*m3*cos(z[1] - z[0])*z[5]**2 - 2*L2*g*m3*sin(z[0]) + Lb**2*m_b*sin(2*z[0])*z[4]**2 - 2*Lb*g*m_b*sin(z[0])))/(2*L3*(-3*L1**2*m1*m3*sin(z[1])**2*cos(z[2] + z[0])**2 + 3*L1**2*m1*(m1 + m3 + m_b)*cos(z[2] + z[0])**2 - 6*L1*L2*m1*m3*sin(z[1] - z[0])*sin(z[1])*cos(z[2] + z[0])*cos(z[2]) + 6*L1*m1*(-L1*m1 + L2*m3 + Lb*m_b)*cos(z[2] + z[0])*cos(z[2])*cos(z[0]) - 3*L2**2*m1*m3*sin(z[1] - z[0])**2*cos(z[2])**2 + 3*L2**2*m3*(m1 + m3 + m_b)*sin(z[1] - z[0])**2 - 6*L2*m3*(-L1*m1 + L2*m3 + Lb*m_b)*sin(z[1] - z[0])*sin(z[1])*cos(z[0]) + m1*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*cos(z[2])**2 + m3*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*sin(z[1])**2 - (m1 + m3 + m_b)*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2) + 3*(-L1*m1 + L2*m3 + Lb*m_b)**2*cos(z[0])**2))
    dz6 = (-2*m3*(L2*cos(z[1] - z[0])*z[4]**2 + g*cos(z[1]))*(3*L1*L2*(m1 + m3 + m_b)*sin(z[1] - z[0])*cos(z[2] + z[0]) - 3*L1*(-L1*m1 + L2*m3 + Lb*m_b)*sin(z[1])*cos(z[2] + z[0])*cos(z[0]) + 3*L2*(-L1*m1 + L2*m3 + Lb*m_b)*sin(z[1] - z[0])*cos(z[2])*cos(z[0]) - (3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*sin(z[1])*cos(z[2])) + 2*(L1*sin(z[2] + z[0])*z[4]**2 - g*sin(z[2]))*(3*L2**2*m3*(m1 + m3 + m_b)*sin(z[1] - z[0])**2 - 6*L2*m3*(-L1*m1 + L2*m3 + Lb*m_b)*sin(z[1] - z[0])*sin(z[1])*cos(z[0]) + m3*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*sin(z[1])**2 - (m1 + m3 + m_b)*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2) + 3*(-L1*m1 + L2*m3 + Lb*m_b)**2*cos(z[0])**2) - 3*(L1*m3*sin(z[1])**2*cos(z[2] + z[0]) - L1*(m1 + m3 + m_b)*cos(z[2] + z[0]) + L2*m3*sin(z[1] - z[0])*sin(z[1])*cos(z[2]) + (L1*m1 - L2*m3 - Lb*m_b)*cos(z[2])*cos(z[0]))*(2*L1*L4*m1*sin(z[2] + z[0])*z[6]**2 + 2*L1*g*m1*sin(z[0]) - 2*L2*L3*m3*cos(z[1] - z[0])*z[5]**2 - 2*L2*g*m3*sin(z[0]) + Lb**2*m_b*sin(2*z[0])*z[4]**2 - 2*Lb*g*m_b*sin(z[0])) - 2*(3*L1*L2*m3*sin(z[1] - z[0])*sin(z[1])*cos(z[2] + z[0]) - 3*L1*(-L1*m1 + L2*m3 + Lb*m_b)*cos(z[2] + z[0])*cos(z[0]) + 3*L2**2*m3*sin(z[1] - z[0])**2*cos(z[2]) - (3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*cos(z[2]))*(L1*m1*sin(z[0])*z[4]**2 - L2*m3*sin(z[0])*z[4]**2 + L3*m3*cos(z[1])*z[5]**2 + L4*m1*sin(z[2])*z[6]**2 - Lb*m_b*sin(z[0])*z[4]**2))/(2*L4*(-3*L1**2*m1*m3*sin(z[1])**2*cos(z[2] + z[0])**2 + 3*L1**2*m1*(m1 + m3 + m_b)*cos(z[2] + z[0])**2 - 6*L1*L2*m1*m3*sin(z[1] - z[0])*sin(z[1])*cos(z[2] + z[0])*cos(z[2]) + 6*L1*m1*(-L1*m1 + L2*m3 + Lb*m_b)*cos(z[2] + z[0])*cos(z[2])*cos(z[0]) - 3*L2**2*m1*m3*sin(z[1] - z[0])**2*cos(z[2])**2 + 3*L2**2*m3*(m1 + m3 + m_b)*sin(z[1] - z[0])**2 - 6*L2*m3*(-L1*m1 + L2*m3 + Lb*m_b)*sin(z[1] - z[0])*sin(z[1])*cos(z[0]) + m1*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*cos(z[2])**2 + m3*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*sin(z[1])**2 - (m1 + m3 + m_b)*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2) + 3*(-L1*m1 + L2*m3 + Lb*m_b)**2*cos(z[0])**2))
    dz7 = (-2*m1*(L1*sin(z[2] + z[0])*z[4]**2 - g*sin(z[2]))*(3*L1*L2*m3*sin(z[1] - z[0])*sin(z[1])*cos(z[2] + z[0]) - 3*L1*(-L1*m1 + L2*m3 + Lb*m_b)*cos(z[2] + z[0])*cos(z[0]) + 3*L2**2*m3*sin(z[1] - z[0])**2*cos(z[2]) - (3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*cos(z[2])) + 2*m3*(L2*cos(z[1] - z[0])*z[4]**2 + g*cos(z[1]))*(3*L1**2*m1*sin(z[1])*cos(z[2] + z[0])**2 + 3*L1*L2*m1*sin(z[1] - z[0])*cos(z[2] + z[0])*cos(z[2]) + 3*L2*(-L1*m1 + L2*m3 + Lb*m_b)*sin(z[1] - z[0])*cos(z[0]) - (3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*sin(z[1])) - 3*(L1*m1*cos(z[2] + z[0])*cos(z[2]) - L2*m3*sin(z[1] - z[0])*sin(z[1]) + (-L1*m1 + L2*m3 + Lb*m_b)*cos(z[0]))*(2*L1*L4*m1*sin(z[2] + z[0])*z[6]**2 + 2*L1*g*m1*sin(z[0]) - 2*L2*L3*m3*cos(z[1] - z[0])*z[5]**2 - 2*L2*g*m3*sin(z[0]) + Lb**2*m_b*sin(2*z[0])*z[4]**2 - 2*Lb*g*m_b*sin(z[0])) + (L1*m1*sin(z[0])*z[4]**2 - L2*m3*sin(z[0])*z[4]**2 + L3*m3*cos(z[1])*z[5]**2 + L4*m1*sin(z[2])*z[6]**2 - Lb*m_b*sin(z[0])*z[4]**2)*(6*L1**2*m1*cos(z[2] + z[0])**2 - 6*L1**2*m1 - 2*L1**2*m_b + 2*L1*L2*m_b + 6*L2**2*m3*sin(z[1] - z[0])**2 - 6*L2**2*m3 - 2*L2**2*m_b - 3*Lb**2*m_b*cos(2*z[0]) - 3*Lb**2*m_b))/(2*(-3*L1**2*m1*m3*sin(z[1])**2*cos(z[2] + z[0])**2 + 3*L1**2*m1*(m1 + m3 + m_b)*cos(z[2] + z[0])**2 - 6*L1*L2*m1*m3*sin(z[1] - z[0])*sin(z[1])*cos(z[2] + z[0])*cos(z[2]) + 6*L1*m1*(-L1*m1 + L2*m3 + Lb*m_b)*cos(z[2] + z[0])*cos(z[2])*cos(z[0]) - 3*L2**2*m1*m3*sin(z[1] - z[0])**2*cos(z[2])**2 + 3*L2**2*m3*(m1 + m3 + m_b)*sin(z[1] - z[0])**2 - 6*L2*m3*(-L1*m1 + L2*m3 + Lb*m_b)*sin(z[1] - z[0])*sin(z[1])*cos(z[0]) + m1*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*cos(z[2])**2 + m3*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2)*sin(z[1])**2 - (m1 + m3 + m_b)*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*cos(z[0])**2) + 3*(-L1*m1 + L2*m3 + Lb*m_b)**2*cos(z[0])**2))
    return np.array([dz0,dz1,dz2,dz3,dz4,dz5,dz6,dz7])

def TrebOnWheels(x):
    global g
    global L2
    global L3
    global m3
    global L1
    global L4
    global Lb
    global m_b
    global endtime
    global dt
    global t,Sx
    global x1,y1,x2,y2,x3,y3,x4,y4,h

    g = 9.81
    #m1 =
    m3 = 1
    L1 = 1
    L2 = x[0]
    L3 = x[1]
    Lb = (L2-L1)/2

    m_b = 1
    h = -np.cos(45*pi/180)*L2
    L4 = - L1 - h
    endtime = 0.7
    dt = int(4*m1+600)#3000
    t = np.linspace(0,endtime,dt)
    zinit = np.array([45*pi/180, 1*pi/180, 1*pi/180, 0.1, 0, 0, 0, 0])
    z = odeint(deriv, zinit, t)

    x5 = z[:,3]
    y5 = np.zeros(dt)
    y0 = np.zeros(dt)
    xb = x5 - Lb*np.sin(z[:,0])
    yb = y5 - Lb*np.cos(z[:,0])
    x1 = x5 + L1*np.sin(z[:,0])
    y1 = y5 + L1*np.cos(z[:,0])
    x2 = x5 - L2*np.sin(z[:,0])
    y2 = y0 - L2*np.cos(z[:,0])
    x3 = x2 + L3*np.cos(z[:,1])
    y3 = y2 - L3*np.sin(z[:,1])
    x4 = x1 + L4*np.sin(z[:,2])
    y4 = y1 - L4*np.cos(z[:,2])

    rangethrown = 0
    Sx = []
    #z[0] = theta
    #z[1] = alpha
    #z[2] = phi
    #z[3] = x0
    #z[4] = theta_dot
    #z[5] = alpha_dot
    #z[6] = phi_dot
    #z[7] = x0_dot
    #ThrowRange(theta,thetadot,alpha,alphadot,x0_dot):
    for i in range(0,len(t)):
        Sx.append(ThrowRange(z[i,0],z[i,4],z[i,1],z[i,5],z[i,7]))
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
    return -rangethrown


size = 50
#massRatioSpace = np.logspace(50,10000,size)
massRatioSpace = np.logspace(2,4,num=size)

L2space = np.zeros(size)
L3space = np.zeros(size)
L4space = np.zeros(size)
rangeSpace = np.zeros(size)
x0 = np.array([3,3])

arraycounter = 0
for i in massRatioSpace:
    ## OPTIMIZER
    global m1
    print "massRatio: %.3f" %(massRatioSpace[arraycounter])

    m1 = massRatioSpace[arraycounter]
    #res = minimize(fatTreb, x0, method='TNC', bounds = ((2,10),(2,10)), options={'maxiter': 300, 'disp': True}) # tol = 1e-8
    res = minimize(TrebOnWheels, x0, method='Nelder-Mead', bounds = ((1/np.cos(np.pi/4),30),(1.5,30)), options={'maxiter': 300, 'disp': True}) # xtol = 1e-8'maxiter': 300,
    print "dt: %d" % dt
    L2space[arraycounter] = res.x[0]
    L3space[arraycounter] = res.x[1]
    L4space[arraycounter] = L4
    rangeSpace[arraycounter] = -TrebOnWheels([res.x[0],res.x[1]])
    print "For L2: %.3f L3: %.3f L4: %.3f Range: %.3f" % (L2space[arraycounter], L3space[arraycounter], L4space[arraycounter], rangeSpace[arraycounter])
    arraycounter += 1
    x0 = np.array([res.x[0],res.x[1]])
    print(res.x)
# This function automatically updates the input variables.
#print "Length ratios:"
#print "L1/L1: %.3f L2/L1: %.3f L3/L1 %.3f" % (L1/L1, res.x[0]/L1, res.x[1]/L1)

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
# Yankee Seige 10bs pumpkin 21,000lbs CW = 2100 m1/m2
#plt.semilogx([2100,2100],[0,1],color='r',linewidth = 0.5,linestyle = '--')
# American Chucker CW 1400, 10lb pumpkin = 140
#plt.semilogx([140,140],[0,1],color='r',linewidth = 0.5,linestyle = '--')
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.semilogx(massRatioSpace,L2space, color="r", marker="o", markersize = 2,markerfacecolor='None',markeredgecolor = 'r', linestyle = "None", label=r'$\frac{L_{2}}{L_{1}}$')
line2, = plt.semilogx(massRatioSpace,L3space, color="k", marker="+", markersize = 2,markerfacecolor='w',markeredgecolor = 'b', linestyle = "None", label=r'$\frac{L_{3}}{L_{1}}$')
line3, = plt.semilogx(massRatioSpace,L4space, marker="s", markersize = 2,markerfacecolor='w',markeredgecolor = 'c', linestyle = "None", label=r'$\frac{L_{4}}{L_{1}}$')

# Plot the range on the y-axis two.
ax2 = ax.twinx()
line4, = ax2.loglog(massRatioSpace, rangeSpace, color="g", marker="^", markersize = 2,markerfacecolor='None',markeredgecolor = 'g', linestyle = "None", label=r'Range')

ax2.set_ylabel('Range')

ax.set_xlabel(r'CW/projectile ($m_{1}/m_{2}$) mass ratio')
ax.set_ylabel(r'Arm ratio')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
#plt.legend(h1+h2, l1+l2, loc=0, handler_map={line1: HandlerLine2D(numpoints=1),line2: HandlerLine2D(numpoints=1),line3: HandlerLine2D(numpoints=1),line4: HandlerLine2D(numpoints=1)})

xlimits = plt.xlim(80,13000)

handles, labels = ax.get_legend_handles_labels()
lgd = plt.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5, 1.4), handler_map={line1: HandlerLine2D(numpoints=1),line2: HandlerLine2D(numpoints=1),line3: HandlerLine2D(numpoints=1),line4: HandlerLine2D(numpoints=1)}, ncol = 4)

#ax.grid('on')
#fig.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.tight_layout()
savefig('TrebOnWheelsOptimisation')

# Pickle out all the data for later processing:
import pickle
#combine the massRatioSpace, L2space,L3space,rangeSpace
outData = list([massRatioSpace, L2space, L3space, L4space, rangeSpace])
PIK = "OptimiseTrebOnWheelsModel.dat"
with open(PIK, "wb") as f:
    pickle.dump(outData, f)
with open(PIK, "rb") as f:
    print pickle.load(f)