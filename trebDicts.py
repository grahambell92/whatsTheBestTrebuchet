

import numpy as np
from FATTrebClass import FATTreb
from pendulumTrebClass import pendulumTreb
from standardTrebOnWheelClass2 import standardTrebOnWheels
from standardTrebClass import standardTreb
from murlinTrebClass import murlinTreb, solveAandB
from simplifiedMurlinTrebClass import simpleMurlinTreb
import optimizeAllTrebs

#---- Trebuchet Dictionaries -----------
pendulumDict = {'trebType': 'Pendulum',
           'pickleFileName': 'OptimisePendulumModel.dat',
           'TrebClass': pendulumTreb(name='TestTreb'),
           'm1': 100,
           'm3': 1,
           'zinit' : np.array([45 * np.pi / 180, 0.01]),
           'beamDensity': 1.0,
           'g': 9.81,
           'endtime': 2.5,
           'dt':0.01,
           'x0': np.array([2]),
           'optimisationBounds':((0.8),(10.0)),
           'itemNames' : ['L2', 'Throw angle', 'Range'],
           'primaryPlotItems' :['L2'],
           'primaryPlotLabels' :[r'$\frac{L_{2}}{L_{1}}$'],
           'secondaryPlotItems' : ['Range'],
           'secondaryPlotLabels' : 'Range',
           'optiPlotName' : 'pendulumOptimisation',
           'tabledDataName' : 'pendulumTabulated.csv',
           'energyPlotName' : 'pendulumEnergy'
           }

FATDict = {'trebType': 'FAT',
           'pickleFileName': 'OptimiseFATModel.dat',
           'TrebClass': FATTreb,
           #'m1': 100,
           'm3': 1.0,
           'zinit' : np.array([45.0 * np.pi / 180, 0.01 * np.pi / 180, 0.0, 0.0]),
           'beamDensity': 1.0,
           'g': 9.81,
           'endtime': 2.0,
           'dt': 0.01,
           'x0': np.array([3.0, 3.0]),
           'optimisationBounds':((0.4,12),(0.4,12.0)),
           'itemNames' : ['L2', 'L3', 'Throw angle', 'Energy Efficiency', 'Range'],
           'primaryPlotItems' :['L2', 'L3'],
           'primaryPlotLabels' :[r'$\frac{L_{2}}{L_{1}}$', r'$\frac{L_{3}}{L_{1}}$'],
           'secondaryPlotItems' : ['Range'],
           'secondaryPlotLabels' : 'Range',
           'optiPlotName' : 'FATOptimisation',
           'tabledDataName' : 'FATTabulated.csv',
           'energyPlotName' : 'FATEnergy'
           }

standardDict = {'trebType': 'Standard',
           'pickleFileName': 'OptimiseStandardModel.dat',
           'TrebClass': standardTreb,
           #'m1': 100,
           'm3': 1,
           'zinit' : np.array([45*np.pi/180, 1*np.pi/180, 1*np.pi/180, 0, 0, 0]),
           'beamDensity': 1.0,
           'g': 9.81,
           'endtime': 2.5,
           'dt': 0.01,
           'x0': np.array([2,2]),
           'optimisationBounds':((0.6,4.0), (0.6,4.0)),
           'itemNames' : ['L2', 'L3', 'Throw angle', 'Energy Efficiency', 'Range'],
           'primaryPlotItems' :['L2', 'L3'],
           'primaryPlotLabels' :[r'$\frac{L_{2}}{L_{1}}$', r'$\frac{L_{3}}{L_{1}}$'],
           'secondaryPlotItems' : ['Range'],
           'secondaryPlotLabels' : 'Range',
           'optiPlotName' : 'standardOptimisation',
           'tabledDataName' : 'StandardTabulated.csv',
           'energyPlotName' : 'standardEnergy'
           }

standardWheeledDict = {'trebType': 'StandardWheeled',
           'pickleFileName': 'OptimiseStandardWheeledModel.dat',
           'TrebClass': standardTrebOnWheels,
           'm1': 100,
           'm3': 1,
           'zinit' : np.array([45*np.pi/180, 1*np.pi/180, 1*np.pi/180, 0.1, 0, 0, 0, 0]),
           'beamDensity': 1.0,
           'g': 9.81,
           'endtime': 1.5,
           'dt': 0.01,
           'x0': np.array([2,2]),
           'optimisationBounds':((0.6,2.5),(0.6,2.5)),
           'itemNames' : ['L2', 'L3', 'Throw angle', 'Energy Efficiency', 'Range'],
           'primaryPlotItems' :['L2', 'L3'],
           'primaryPlotLabels' :[r'$\frac{L_{2}}{L_{1}}$', r'$\frac{L_{3}}{L_{1}}$'],
           'secondaryPlotItems' : ['Range'],
           'secondaryPlotLabels' : 'Range',
           'optiPlotName' : 'standardWheelsOptimisation',
           'tabledDataName' : 'StandardWheeledTabulated.csv',
           'energyPlotName' : 'standardWheeledEnergy'
           }

murlinDict = {'trebType': 'Murlin',
           'pickleFileName': 'murlinModel.dat',
           'TrebClass': murlinTreb,
           'm1': 100,
           'm3': 1,
           'zinit' : np.array([0.01 * np.pi / 180, 0.1 * np.pi / 180, 0.0, 0.0]),
           'beamDensity': 1.0,
           'g': 9.81,
           'endtime': 2.0,
           'dt': 0.01,
           'x0': np.array([2, 2, 0.4]),
           'x4offset': 0.1,
           'optimisationBounds':((0.6,2.5),(0.6,2.5), (0.3, 0.97)),
           'itemNames' : ['L2', 'L3', 'a', 'b', 'Throw angle', 'Energy Efficiency', 'Range'],
           'primaryPlotItems' :['L2', 'L3'],
           'primaryPlotLabels' :[r'$\frac{L_{2}}{L_{1}}$', r'$\frac{L_{3}}{L_{1}}$'],
           'secondaryPlotItems' : ['Range'],
           'secondaryPlotLabels' : 'Range',
           'optiPlotName' : 'murlinOptimisation',
           'tabledDataName' : 'MurlinTabulated.csv',
           'energyPlotName' : 'murlinEnergy'
           }

simpleMurlinDict = {'trebType': 'Murlin Simple',
           'pickleFileName': 'simpleMurlinModel.dat',
           'TrebClass': simpleMurlinTreb,
           'm1': 100,
           'm3': 1,
           'zinit' : np.array([0.0 * np.pi / 180, 0.0 * np.pi / 180, 0.0, 0.0]),
           'beamDensity': 1.0,
           'g': 9.81,
           'endtime': 2.0,
           'dt': 0.01,
           'x0': np.array([2, 2, 1.4]),
           'optimisationBounds':((0.6,2.5), (0.6,2.5), (0.3, 1.5)),
           'itemNames' : ['L2', 'L3', 'a', 'b', 'Throw angle', 'Energy Efficiency', 'Range'],
           'primaryPlotItems' :['L2', 'L3'],
           'primaryPlotLabels' :[r'$\frac{L_{2}}{L_{1}}$', r'$\frac{L_{3}}{L_{1}}$'],
           'secondaryPlotItems' : ['Range'],
           'secondaryPlotLabels' : 'Range',
           'optiPlotName' : 'simpleMurlinOptimisation',
           'tabledDataName' : 'simpleMurlinTabulated.csv',
           'energyPlotName' : 'murlinSimpleEnergy'
           }