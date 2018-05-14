# Created to practice animation with


import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from trebDicts import FATDict, murlinDict, standardDict, standardWheeledDict, pendulumDict, simpleMurlinDict
import os

trebDicts = [pendulumDict, FATDict, standardDict, standardWheeledDict, murlinDict, simpleMurlinDict]
trebDicts = [trebDicts[5]]

for trebDict in trebDicts:
    pickleFileName = trebDict['pickleFileName']
    plotLabel = trebDict['trebType']
    print('Reading Name', pickleFileName)
    data = pd.read_pickle(pickleFileName)
    massRatioSpace = data.index.values

    # Create treb instances
    trebs = [trebDict['TrebClass']() for i in range(len(data))]

    # Run all trebs from optimisation data
    zinit = trebDict['zinit']
    # Get the columns of the dataframe.
    # Everything except throw angle, throw range put in.
    for treb, (index, row), massRatio in zip(trebs, data.iterrows(), massRatioSpace):
        if trebDict['trebType'] == murlinDict['trebType']:
            a = row['a']
            b = row['b']
            x4 = a + trebDict['x4offset']
            L2 = row['L2']
            L3 = row['L3']
            range, throwAngle = treb.runTreb(L2=L2, L3=L3, a=a, b=b, x4=x4, m1=massRatio, endtime=2.0, zinit=zinit)

        if trebDict['trebType'] == simpleMurlinDict['trebType']:
            a = row['a']
            b = row['b']
            L2 = row['L2']
            L3 = row['L3']
            range, throwAngle = treb.runTreb(L2=L2, L3=L3, a=a, m1=massRatio, endtime=2.0, zinit=zinit)

        else:
            L2 = row['L2']
            L3 = row['L3']
            range, throwAngle = treb.runTreb(L2=L2, L3=L3, m1=massRatio, endtime=2.0, zinit=zinit)
        print(L2, L3, massRatio)

    # Create Plots
    rows = 5
    cols = 6
    totalSize = rows*cols

    for index, time in enumerate(trebs[0].t):
        fig = plt.figure()#figsize=(8,8))
        fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)

        for treb, (rowi, coli) in zip(trebs, itertools.product(np.arange(rows), np.arange(cols))):
            ax = axs[rowi, coli]
            ax = treb.plotTreb(ax, index)
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.set_aspect('equal')
            ax.set_title('{0:.2f}'.format(treb.m1), fontsize=6, y=-0.01)
        fig.suptitle(trebDict['trebType'])

        fig.subplots_adjust(wspace=0, hspace=0.0)
        #fig.tight_layout()
        saveFolder = './Animations/' + trebDict['trebType'] + '/'
        os.makedirs(saveFolder, exist_ok=True)
        fig.savefig(saveFolder + 'Frame' + str(index) + '.png', dpi=500)
        print('index', index)
        plt.close()
