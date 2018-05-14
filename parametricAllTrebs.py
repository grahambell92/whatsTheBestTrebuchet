# Created by Graham Bell 21/07/2016
# This script is an abstraction of the three models that will have
# parametric contours applied.
# The parametric contours require only 2dof optimisation.

import latexPlotTools as lpt
import numpy as np
import matplotlib.pyplot as plt
from FATTrebClass import FATTreb
from standardTrebOnWheelClass2 import standardTrebOnWheels
from standardTrebClass import standardTreb
import pickle

def doParameterStudy(useDict, computeParameterStudy, pickleFileName, plotName, numPoints = 30):
    if computeParameterStudy is True:
        trebClass = useDict[
            'TrebClass']  # FATTreb('TestTreb') # Replace to either standardTreb or standardTreb on Wheels
        zinit = useDict['zinit']
        m1 = useDict['m1']
        m3 = useDict['m3']
        endtime = 2.0
        dt = 0.01
        size = numPoints

        rangeSpace = np.zeros((size, size))
        # L2 loop
        L2s = np.linspace(useDict['L2sMin'], useDict['L2sMax'], size)
        L3s = np.linspace(useDict['L3sMin'], useDict['L3sMax'], size)
        L2Space, L3Space, rangeSpace = np.zeros(size ** 2), np.zeros(size ** 2), np.zeros(size ** 2)

        # ! itertools! who cares about the order!
        # Look how much I've improved in Python!
        import itertools
        for index, [L2, L3] in enumerate(itertools.product(L2s, L3s)):
            throwRange, throwAngle = trebClass.runTreb(m1=m1, m3=m3, L2=L2, L3=L3, endtime=endtime, dt=dt, zinit=zinit)
            L2Space[index] = L2
            L3Space[index] = L3
            rangeSpace[index] = throwRange
            print('Throw:', index, '/', size ** 2 - 1, 'L2:', L2, "L3:", L3, 'Throw range:', throwRange)

        # Pickle out the data
        outData = [L2Space, L2s, L3Space, L3s, rangeSpace, size]
        with open(pickleFilename, "wb") as f:
            print('Written data to:', pickleFilename)
            pickle.dump(outData, f)

    else:
        with open(pickleFilename, "rb") as f:
            [L2Space, L2s, L3Space, L3s, rangeSpace, size] = pickle.load(f)
            print('Loaded pickle data from:', pickleFilename)
    # --------------------------------------------------------------------------

    # Find the L2,L3 of the maximum range
    maxRangeIndex = np.nanargmax(rangeSpace)
    L2Max, L3Max, rangeMax = L2Space[maxRangeIndex], L3Space[maxRangeIndex], np.nanmax(rangeSpace)
    print('L2Max:', L2Max, 'L3Max:', L3Max, 'rangeMax:', rangeMax)
    fig, ax = lpt.newfig(0.6)
    plt.subplot(1, 1, 1)
    levels = np.linspace(np.nanmin(rangeSpace), np.nanmax(rangeSpace) + 0.01, size, endpoint=True)
    patches = []

    plt.plot(L2Space, L3Space, 'k.', markersize=0.35)
    # Draw a plot point on the plot for the highest point.
    # plt.scatter(L3Space[j],L2Space[i],s = 2.0, color = 'r', lw = 0)
    plt.plot(L2Max, L3Max, 'r.', markersize=3.0)

    # Plot 3 times reduces the white lines around each contour level
    # Reshape rangeSpace into a size by size 2d array
    rangeSpace = np.reshape(rangeSpace, (size, size))
    for i in range(3):
        CS = plt.contourf(L2s, L3s, rangeSpace.T, cmap='viridis', levels=levels, linestyle='None')

    cbar = plt.colorbar(ticks=np.linspace(np.nanmin(rangeSpace), np.nanmax(rangeSpace) + 0.01, 6, endpoint=True))

    cbar.ax.set_ylabel('Range (m)')
    ax.set_ylabel(r'$\frac{L_{3}}{L_{1}}$')
    ax.set_xlabel(r'$\frac{L_{2}}{L_{1}}$')

    plt.tight_layout()
    lpt.savefig(plotName)
    #plt.savefig(plotName + '.png', dpi=300)
    plt.clf()

#---- Trebuchet Dictionaries -----------
FATDict = {'pickleFileName': 'ParametricFATTreb.dat',
           'plotName': 'ParametricFATContour',
           'TrebClass': FATTreb('TestTreb'),
           'L2sMin': 0.5,
           'L2sMax': 8,
           'L3sMin': 0.5,
           'L3sMax': 8,
           'm1': 100,
           'm3': 1,
           'zinit' : np.array([45.0 * np.pi / 180, 0.1 * np.pi / 180, 0.0, 0.0])
           }
standardDict = {'pickleFileName': 'ParametricStandardTreb.dat',
           'plotName': 'ParametricStandardTrebContour',
           'TrebClass': standardTreb('TestTreb'),
           'L2sMin': 0.5,
           'L2sMax': 8,
           'L3sMin': 0.5,
           'L3sMax': 8,
           'm1': 100,
           'm3': 1,
           'zinit': np.array([45*np.pi/180, 1*np.pi/180, 1*np.pi/180,0, 0, 0])
           }
wheeledStandardDict = {'pickleFileName': 'ParametricWheeledStandardTreb.dat',
           'plotName': 'ParametricWheeledStandardContour',
           'TrebClass': standardTrebOnWheels('TestTreb'),
           'L2sMin': 1+np.sin(np.pi/4) + 0.2,
           'L2sMax': 8,
           'L3sMin': 0.5,
           'L3sMax': 8,
           'm1': 100,
           'm3': 1,
           'zinit': np.array([45 * np.pi / 180, 1 * np.pi / 180, 1 * np.pi / 180, 0.1, 0, 0, 0, 0])
           }
# Dictionary containing all trebuchets
trebuchets = {'FAT': FATDict, 'standard': standardDict, 'wheeledStandard': wheeledStandardDict}

if __name__ == '__main__':

    # ---------Input----------
    computeParameterStudy = True
    trebDict = trebuchets['wheeledStandard']
    pickleFilename = trebDict['pickleFileName']
    plotName = trebDict['plotName']
    # ------------------------
    doParameterStudy(useDict=trebDict, computeParameterStudy=computeParameterStudy, pickleFileName=pickleFilename, plotName=plotName)
