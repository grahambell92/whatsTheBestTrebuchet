
# Script to plot the
import itertools
import latexPlotTools as lpt
from trebDicts import pendulumDict, FATDict, standardDict, standardWheeledDict, murlinDict, simpleMurlinDict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import trebuchetTools

if __name__ == '__main__':

    #-------------Input----------
    trebDicts = [FATDict, standardDict, standardWheeledDict, simpleMurlinDict]
    #----------------------------

    for trebDict in trebDicts:
        colorCycler = itertools.cycle(lpt.tableau10)
        markerCycler = itertools.cycle(lpt.markers)

        pickleFileName = trebDict['pickleFileName']
        plotLabel = trebDict['trebType']
        print('Reading Name', pickleFileName)
        data = pd.read_pickle(pickleFileName)
        massRatioSpace = data.index.values

        # Create treb instances
        treb = trebDict['TrebClass']()

        # Run all trebs from optimisation data
        zinit = trebDict['zinit']

        rowToPlot = 12

        row = data.iloc[rowToPlot]
        massRatio = data.index.values[rowToPlot]

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


        fig, ax = lpt.newfig(width=0.65)
        CWEnergy = 9.81 * massRatio * (1.0 + np.sin(zinit[0]))
        TKE = treb.totalKineticEnergy()
        TPE = treb.totalPotentialEnergy()
        projPE = trebuchetTools.projPE(mass=treb.m3, yPos=treb.y3, g=treb.g, refHeight=treb.h)
        projKE = trebuchetTools.projKE(mass=treb.m3, xDot=treb.x3Dot, yDot=treb.y3Dot)

        ax.plot(treb.t, TKE, color=next(colorCycler), label='TKE')
        ax.plot(treb.t, TPE, color=next(colorCycler), label='TPE')
        ax.plot(treb.t, projKE, color=next(colorCycler), label='Projectile KE')
        ax.plot(treb.t, projPE, color=next(colorCycler), label='Projectile PE')
        CWLineColor = next(colorCycler)
        ax.axhline(y=CWEnergy, color=CWLineColor, linestyle='-', dashes=[3,1])
        ax.text(s='Total CW potential energy', color=CWLineColor, x=0.05, y=CWEnergy + 50)
        ax.axvline(x=treb.t[treb.maxThrowIndex], color=next(colorCycler), linestyle='-', linewidth=0.8, ymin=0, ymax=CWEnergy)
        ax.set_ylabel('Energy (J)')
        ax.set_xlabel('Time (sec)')
        ax.set_ylim(ymin=0.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.legend()
        plt.tight_layout()
        lpt.savefig(trebDict['energyPlotName'])
        plt.clf()
        print(CWEnergy)
