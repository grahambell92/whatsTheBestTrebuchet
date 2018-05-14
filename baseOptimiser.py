# Optimise all trebuchets
# Written by Graham Bell 19/06/2016

import numpy as np

from scipy.optimize import fsolve, minimize
import pickle
import latexPlotTools as lpt
from matplotlib.legend_handler import HandlerLine2D
from FATTrebClass import FATTreb
from standardTrebOnWheelClass import standardTrebOnWheels
from standardTrebClass import standardTreb
from murlinTrebClass import murlinTreb, solveAandB



#---- Trebuchet Dictionaries -----------
FATDict = {'pickleFileName': 'OptimiseFATModel.dat',
           'plotName': 'FATOptimise.dat',
           'TrebClass': FATTreb(name='TestTreb'),
           'm1': 100,
           'm3': 1,
           'zinit' : np.array([45.0 * np.pi / 180, 0.1 * np.pi / 180, 0.0, 0.0]),
           'beamDensity': 1.0,
           'g': 9.81,
           'endtime': 2.5,
           'dt':0.01,
           'throwFunc': throwFATTreb,
           'x0': np.array([2,2]),
           'optimisationBounds':((1.0,20),(1.0,20.0))
           'numResultVars': 3
           }

#-------------Input----------
computeOptimisation = True
trebDict = FATDict
pickleFilename = trebDict['pickleFileName']
#----------------------------


if computeOptimisation is True:
    zinit = trebDict['zinit']
    endtime = trebDict['endtime']
    dt = trebDict['dt']

    testTreb = trebDict['TrebClass'] #  murlinTreb(name="TestTrebOnWheels", beamDensity = beamDensity, g = 9.81)
    testTreb.beamDensity =  trebDict['beamDensity']
    testTreb.g = trebDict['g']

    # Find Optimum of mass ratios m1/m2 from 1:50 to 1:10000 logarithmic functon?
    size = 50

    massRatioSpace = np.logspace(np.log10(20),3,num=size)
    #massRatioSpace = np.array([100])

    #L2space = np.zeros(size)
    #L3space = np.zeros(size)
    #aSpace = np.zeros(size)
    #bSpace = np.zeros(size)
    #rangeSpace = np.zeros(size)
    #for i in range(trebDict['numOptimisationVars']):
    # Initialise the result array
    resultSpace = np.zeros((size, trebDict['numResultVars']))
    x0 = trebDict['x0']
    for index, value in enumerate(massRatioSpace):
        ## OPTIMIZER
        print("massRatio:", value)
        m1 = value
        res = minimize(trebDict['throwFunc'], x0, method='Nelder-Mead',bounds = trebDict['optimisationBounds'], tol = 1e-3, options={'maxiter': 60, 'disp': True})
        for varIndex, result in enumerate(res.x):
            resultSpace[index,varIndex] = result
            # Put in the range again by performing the function
            resultSpace[index, -1] = -trebDict['throwFunc'](res.x)
        #L2space[index] = res.x[0]
        #L3space[index] = res.x[1]
        #aSpace[index] = res.x[2]
        # Setting the b value for the murlin treb.
        #x4 = res.x[2] + x4offset
        #bSpace[index] = fsolve(solveAandB, 0.1, args=(res.x[2], x4))[0]
        #rangeSpace[index] = -throwMurlinTreb([res.x[0],res.x[1],res.x[2]])
        #x0 = np.array([res.x[0],res.x[1], res.x[2]])
        print(res.x)

    #combine the massRatioSpace, L2space,L3space,rangeSpace
    outData = [massRatioSpace, L2space, L3space, aSpace, bSpace, rangeSpace]
    with open(pickleFilename, "wb") as f:
        pickle.dump(outData, f)

else:
    with open(pickleFilename, "rb") as f:
        [massRatioSpace, L2space, L3space, aSpace, bSpace, rangeSpace] = pickle.load(f)
        print('Loaded pickle data.')



# Do plotting via latexPlotTools
import matplotlib.pyplot as plt
fig, ax = lpt.newfig(0.6)
#lpt.subplot(2,1,1)

line1, = plt.semilogx(massRatioSpace,L2space, color="r", marker="o", markersize = 2,markerfacecolor='None',markeredgecolor = 'r', linestyle = "None", label=r'$\frac{L_{2}}{L_{1}}$')
line2, = plt.semilogx(massRatioSpace,L3space, color="b", marker="+", markersize = 2,markerfacecolor='w', linestyle = "None", label=r'$\frac{L_{3}}{L_{1}}$')

# Plot the range on the y-axis two.
ax2 = ax.twinx()
line3, = ax2.loglog(massRatioSpace, rangeSpace, color="g", marker="^", markersize = 2,markerfacecolor='None',markeredgecolor = 'g', linestyle = "None", label=r'Range')
#ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.set_ylabel('Range')

#ax2.set_ylim(0.007,2.000)

ax.set_xlabel(r'CW/projectile ($m_{1}/m_{2}$) mass ratio')
ax.set_ylabel(r'Arm ratio')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
plt.legend(h1+h2, l1+l2, loc=0, handler_map={line1: HandlerLine2D(numpoints=1),line2: HandlerLine2D(numpoints=1),line3: HandlerLine2D(numpoints=1)})
xlimits = plt.xlim(10,2000)
plt.tight_layout()
lpt.savefig('MurlinOptimisation')


# Plot a and range on the other graph
fig, ax = lpt.newfig(0.6)
#lpt.subplot(2,1,1)

line1, = plt.semilogx(massRatioSpace,aSpace, color="r", marker="o", markersize = 2,markerfacecolor='None',markeredgecolor = 'r', linestyle = "None", label = r'$a$~value')
#plt.ylim(0.97*np.min(aSpace), 1.03*np.max(aSpace))
line2, = plt.semilogx(massRatioSpace,bSpace, color="b", marker="+", markersize = 2,markerfacecolor='w', linestyle = "None", label = r'$b$~value')
# Plot the range on the y-axis two.
ax2 = ax.twinx()
line3, = ax2.loglog(massRatioSpace, rangeSpace, color="g", marker="^", markersize = 2,markerfacecolor='None',markeredgecolor = 'g', linestyle = "None", label = 'range')
#ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.set_ylabel('Range')

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
#ax2.set_ylim(0.007,2.000)

ax.set_xlabel(r'CW/projectile ($m_{1}/m_{2}$) mass ratio')
ax.set_ylabel(r'Cam value')
xlimits = plt.xlim(10,2000)
plt.legend(h1+h2, l1+l2, loc=0, handler_map={line1: HandlerLine2D(numpoints=1),line2: HandlerLine2D(numpoints=1),line3: HandlerLine2D(numpoints=1)})
plt.tight_layout()
lpt.savefig('MurlinOptimisationpt2')