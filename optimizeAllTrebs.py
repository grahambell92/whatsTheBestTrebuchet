# Murlin Trebuchet Optimisation
# Written by Graham Bell 19/06/2016

import numpy as np
from scipy.optimize import fsolve, differential_evolution
import pickle
import latexPlotTools as lpt
import trebDicts
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import trebuchetTools

def throwPendulumTreb(x, returnThrowAngle=False):
    [L2] = x
    throwRange, throwAngle = testTreb.runTreb(m1=m1, m2=m3, L2 = L2,endtime = endtime, dt = dt, zinit=zinit)
    print("Optimizer Trying:", 'L2:', L2, 'Range:',throwRange, 'throw angle:', np.rad2deg(throwAngle))
    # If something else is returned as type then it's to return both range and throw angle.
    # The optimiser will not provide anything for type
    if returnThrowAngle is True:
        return -throwRange, throwAngle
    else:
        return -throwRange

def throwMurlinTreb(x, returnThrowAngle=False):
    [L2, L3, a] = x
    x4 = a + trebDict['x4offset']
    b = fsolve(solveAandB, 0.1, args=(a, x4))[0]
    throwRange, throwAngle = testTreb.runTreb(m1=m1, m3=m3, L2 = L2, L3 = L3, x4 = x4, endtime = endtime, dt = dt, zinit=zinit, a = a, b = b)
    print("Optimizer Trying:", 'a:', a, 'x4:', 'b:', b, 'L2:', L2, 'L3:', L3, "Range:", throwRange, 'throw angle:', np.rad2deg(throwAngle))
    if returnThrowAngle is True:
        return -throwRange, throwAngle
    else:
        return -throwRange

def packThrowResult(resultDataFrame, trebDict, massRatio, optiResult, optiRange, optiThrowAngle, throwEnergyEff):
    if trebDict['trebType'] == trebDicts.FATDict['trebType']:
        optiL2 = optiResult[0]
        optiL3 = optiResult[1]
        results = [optiL2, optiL3, optiThrowAngle, throwEnergyEff, optiRange]
    elif trebDict['trebType'] == trebDicts.pendulumDict['trebType']:
        optiL2 = optiResult[0]
        results = [optiL2, optiThrowAngle, throwEnergyEff, optiRange]

    elif trebDict['trebType'] == trebDicts.standardDict['trebType']:
        optiL2 = optiResult[0]
        optiL3 = optiResult[1]
        results = [optiL2, optiL3, optiThrowAngle, throwEnergyEff, optiRange]

    elif trebDict['trebType'] == trebDicts.standardWheeledDict['trebType']:

        optiL2 = optiResult[0]
        optiL3 = optiResult[1]
        results = [optiL2, optiL3, optiThrowAngle, throwEnergyEff, optiRange]

    elif trebDict['trebType'] == trebDicts.murlinDict['trebType']:
        optiL2 = optiResult[0]
        optiL3 = optiResult[1]
        optiA = optiResult[2]
        # Setting the b value for the murlin treb.
        x4 = optiA + trebDict['x4offset']
        optiB = fsolve(testTreb.solveAandB, 0.1, args=(optiA, x4))[0]
        results = [optiL2, optiL3, optiA, optiB, optiThrowAngle, throwEnergyEff, optiRange]

    elif trebDict['trebType'] == trebDicts.simpleMurlinDict['trebType']:
        optiL2 = res.x[0]
        optiL3 = res.x[1]
        optiA = res.x[2]
        # Setting the b value for the murlin treb.
        optiB = fsolve(testTreb.solveAandB, 0.4, args=(optiA))[0]
        results = [optiL2, optiL3, optiA, optiB, optiThrowAngle, throwEnergyEff, optiRange]

    # Insert the values for all the results here.
    for item, result in zip(trebDict['itemNames'], results):
        resultDataFrame.set_value(massRatio, item, float(result))
    return resultDataFrame

if __name__ == '__main__':

    #-------------Input----------
    computeOptimisation = False
    dicts = [trebDicts.simpleMurlinDict] #[trebDicts.standardWheeledDict]

    #----------------------------
    for trebDict in dicts:

        colorCycler = itertools.cycle(lpt.tableau10)
        markerCycler = itertools.cycle(lpt.markers)
        if computeOptimisation is True:
            print('Starting treb:', trebDict['trebType'])
            testTreb = trebDict['TrebClass']()
            testTreb.zinit = trebDict['zinit']
            testTreb.endtime = trebDict['endtime']
            testTreb.dt = trebDict['dt']

            testTreb.beamDensity = trebDict['beamDensity']
            testTreb.g = trebDict['g']

            # Find Optimum of mass ratios m1/m2 from 1:50 to 1:10000 logarithmic functon?
            size = 30
            massRatioSpace = np.logspace(np.log10(20), 3, num=size)
            #massRatioSpace = [30.0, 40.0]

            # Initialise and store the throw results in a pandas dataframe.
            resultDataFrame = pd.DataFrame(index=massRatioSpace, columns=trebDict['itemNames'])
            resultDataFrame.index.name = 'Mass ratio'
            x0 = trebDict['x0']
            optiBounds = trebDict['optimisationBounds']
            for index, value in enumerate(massRatioSpace):
                ## OPTIMIZER
                print("massRatio:", value)
                print('Bounds Set to:', optiBounds, 'for mass ratio:', value)
                testTreb.m1 = value

                testTreb.m3 = trebDict['m3']

                #res = differential_evolution(trebDict['throwFunc'], disp=True, bounds=optiBounds,
                #                             maxiter=20, tol=0.01, popsize=6, polish=False)
                res = differential_evolution(testTreb.optimiserThrowTreb, disp=True, bounds=optiBounds,
                                             maxiter=20, tol=0.01, popsize=6, polish=False)
                # repeat the throw with res to get the range and throw angle.
                optiRange, optiThrowAngle = testTreb.optimiserThrowTreb(res.x, returnThrowAngle=True)
                # Get optimum energy functions to report energy efficiency.
                TKE = testTreb.totalKineticEnergy()[testTreb.maxThrowIndex]
                TPE = testTreb.totalPotentialEnergy()[testTreb.maxThrowIndex]
                projKE = trebuchetTools.projKE(mass=testTreb.m3, xDot=testTreb.x3Dot, yDot=testTreb.y3Dot)[testTreb.maxThrowIndex]
                throwEfficiency = projKE/(TKE+TPE)
                print('Energy efficiency', throwEfficiency)

                # Now send the result to the result packer to fill-out the variables in the resultDataFrame
                resultDataFrame = packThrowResult(resultDataFrame=resultDataFrame, trebDict=trebDict, massRatio=value,
                                                  optiResult=res.x, optiRange=-optiRange,
                                                  throwEnergyEff=throwEfficiency,
                                                  optiThrowAngle=np.rad2deg(optiThrowAngle))

                spacer = 2.0
                if trebDict['trebType'] == trebDicts.murlinDict['trebType']:
                    optiBounds = ((res.x[0] - 0.3, res.x[0] + spacer), (res.x[1] - 0.3, res.x[1] + spacer), (0.3, 0.97))

                elif trebDict['trebType'] == trebDicts.simpleMurlinDict['trebType']:
                    optiBounds = ((res.x[0] - 0.3, res.x[0] + spacer), (res.x[1] - 0.3, res.x[1] + spacer), (0.3, 1.6))

                elif trebDict['trebType'] == trebDicts.standardWheeledDict['trebType']:
                    print('here in standard')
                    optiBounds = ((res.x[0]-0.3, res.x[0]+spacer), (res.x[1]-0.3, res.x[1]+spacer))
                else:
                    optiBounds = ((res.x[0] - 0.35, res.x[0] + spacer), (res.x[1] - 0.35, res.x[1] + spacer))

                print(resultDataFrame)

            with open(trebDict['pickleFileName'], "wb") as f:
                resultDataFrame.to_pickle(trebDict['pickleFileName'])
                #pickle.dump(resultDataFrame, f)

        else:
            with open(trebDict['pickleFileName'], "rb") as f:
                resultDataFrame = pickle.load(f)
                print(type(resultDataFrame))
                print('Loaded pickle data.')
                print(resultDataFrame)

        # Write out the resulting tabulated data to csv.
        lpt.saveToCSV(trebDict['tabledDataName'], resultDataFrame)

        massRatioSpace = resultDataFrame.index.values

        # Do plotting via latexPlotTools
        # Plot optimisation range plots
        import matplotlib.pyplot as plt
        fig, ax = lpt.newfig(0.6)
        # plot the primary plots
        for plotItem, plotLabel in zip(trebDict['primaryPlotItems'], trebDict['primaryPlotLabels']):
            plt.semilogx(massRatioSpace, resultDataFrame[plotItem], markeredgecolor=next(colorCycler), markerfacecolor='None', markersize=2,
                         marker=next(markerCycler), linestyle="None", label=plotLabel)

        # Plot the range on the y-axis two.
        ax2 = ax.twinx()

        ax2.loglog(massRatioSpace, resultDataFrame['Range'], marker="^", markersize=2,
                   markerfacecolor='None', markeredgecolor=lpt.tableau10[6], linestyle="None", label=trebDict['secondaryPlotLabels'])
        ax2.set_ylabel('Range (m)')
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax2.yaxis.set_ticks_position('right')
        plt.xlim(10, 2000)
        ax2.set_ylim(10,10000)
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel(r'CW/projectile ($m_{1}/m_{2}$) mass ratio')
        ax.set_ylabel(r'Arm ratio')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        handles = h1 + h2
        labels = l1 + l2
        plt.legend(handles, labels, loc=0)


        plt.tight_layout()
        lpt.savefig(trebDict['optiPlotName'])
        plt.clf()

        # Reset colorcycler
        #colorCycler = itertools.cycle(lpt.tableau10)
        if trebDict['trebType'] == trebDicts.murlinDict['trebType'] or trebDict['trebType'] == trebDicts.simpleMurlinDict['trebType']:
            # Plot a and range on the other graph
            fig, ax = lpt.newfig(0.6)

            plt.semilogx(massRatioSpace, resultDataFrame['a'], markeredgecolor = next(colorCycler), marker=next(markerCycler), markersize=2,
                         markerfacecolor='None', linestyle="None", label=r'$a$')
            plt.semilogx(massRatioSpace,resultDataFrame['b'], markeredgecolor = next(colorCycler), marker=next(markerCycler), markersize=2,
                         markerfacecolor='None', linestyle="None", label=r'$b$')

            # Plot the range on the y-axis two.
            ax2 = ax.twinx()
            ax.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax2.yaxis.set_ticks_position('right')
            ax.xaxis.set_ticks_position('bottom')

            ax2.loglog(massRatioSpace, resultDataFrame['Range'], color="g", marker="^", markersize=2 ,markerfacecolor='None',
                       markeredgecolor=lpt.tableau10[6], linestyle="None", label='Range')
            ax2.set_ylabel('Range (m)')
            ax2.set_ylim(10,10000)
            ax.set_xlabel(r'CW/projectile ($m_{1}/m_{2}$) mass ratio')
            ax.set_ylabel(r'Cam value')
            xlimits = plt.xlim(10, 2000)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            handles = h1 + h2
            labels = l1 + l2
            lgnd = plt.legend(handles, labels, loc=7, ncol=3, bbox_to_anchor=(0.85, -0.3))
            plt.tight_layout()
            lpt.savefig(trebDict['optiPlotName'] + '2', bbox_extra_artists=(lgnd,), bbox_inches='tight')
