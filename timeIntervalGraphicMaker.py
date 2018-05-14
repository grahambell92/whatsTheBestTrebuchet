# Short script to utilize the murlin trebuchet class to create the time interval graphic.

# Written by Graham Bell 6th December 2016.

import numpy as np
from trebDicts import simpleMurlinDict, standardDict, FATDict, standardWheeledDict
import matplotlib.pyplot as plt
import latexPlotTools as lpt
import pickle

trebDicts = [standardDict, standardWheeledDict, simpleMurlinDict, FATDict]

for trebDict in trebDicts:

    # Load in the example dataframe
    with open(trebDict['pickleFileName'], "rb") as f:
        resultDataFrame = pickle.load(f)
    # Perform the throw at m1 = 50
    #trebDict = simpleMurlinDict
    # Create instance of the trebuchet.
    treb = trebDict['TrebClass']()

    zinit = trebDict['zinit']

    # take the optimisation data at xth spot.
    #print(resultDataFrame)
    massRatioSpace = resultDataFrame.index.values
    useIndex = 10
    m1 = massRatioSpace[useIndex]

    if trebDict['trebType'] == simpleMurlinDict['trebType']:
        a = resultDataFrame.iloc[useIndex]['a']
        b = resultDataFrame.iloc[useIndex]['b']
        L2 = resultDataFrame.iloc[useIndex]['L2']
        L3 = resultDataFrame.iloc[useIndex]['L3']
        range, throwAngle = treb.runTreb(L2=L2, L3=L3, a=a, m1=m1, endtime=2.0, zinit=zinit)

    else:

        L2 = resultDataFrame.iloc[useIndex]['L2']
        L3 = resultDataFrame.iloc[useIndex]['L3']
        range, throwAngle = treb.runTreb(L2=L2, L3=L3, m1=m1, endtime=2.0, zinit=zinit)
        print(L2, L3, m1)

    if True:
        # Throw step index
        maxThrowIndex = treb.maxThrowIndex
        print('range', range, 'throw Angle', throwAngle, 'maxThrowIndex', maxThrowIndex)

        fig, ax = lpt.newfig(0.8)
        ax = fig.add_subplot(1, 1, 1)
        frameInterval = 5
        numFrames = maxThrowIndex/frameInterval
        frames = np.linspace(0, maxThrowIndex, numFrames, dtype=np.int)
        alphas = np.logspace(-1, 1, numFrames)/10
        alphas[:] = 0.2
        for frame, imageAlpha in zip(frames, alphas):
            ax = treb.plotTreb(ax, i=int(frame), projRadius=0.1, CWRadius=0.2, lineThickness=1.0, beamCentreRadius=0.1,
                               imageAlpha=imageAlpha)
        # Plot the throw freeze point
        ax = treb.plotTreb(ax, i=maxThrowIndex, projRadius=0.1, CWRadius=0.2, lineThickness=1.0, beamCentreRadius=0.1)

        bounds = L2 + L3
        ax.set_xlim(-bounds, bounds)
        ax.set_ylim(-4, bounds)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$(m)')
        ax.set_ylabel(r'$y$(m)')
        plotName = trebDict['trebType'] + 'TimeInterval'
        plt.tight_layout()
        lpt.savefig(plotName)
        plt.close('all')
        #fig.savefig('./murlinTimeInterval.png', dpi=500)
