# Class for the FPendulum Trebuchet
# Created by Graham Bell 08/07/2016

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class pendulumTreb:
    def __init__(self,name= 'treb1', g=9.81,beamDensity = 1.0):
        self.name = name
        self.g = g
    def throwAngle(self, theta):
        return theta - np.pi / 2
    def ThrowRange(self, theta, thetadot):
        x2 = -self.L2 * np.sin(theta)
        y2 = -self.L2 * np.cos(theta)

        Vx = self.L2 * thetadot * np.sin(self.throwAngle(theta))
        Vy = self.L2 * thetadot * np.cos(self.throwAngle(theta))

        Sx = Vx * ((Vy + np.sqrt((Vy ** 2) + (2 * self.g * (y2 - self.h)))) / self.g) + x2
        #print(Sx, np.rad2deg(theta), np.rad2deg(self.throwAngle(theta)))
        if np.isnan(Sx):
            Sx = 0
        # Sx = 2*Vx*Vy/g
        # theta > pi/2 and theta < 3*pi/2 and
        if Vx > 0 and Vy > 0 and Sx > 0:
            #print("Vx: %.3f" % Vx)
            #print("Vy: %.3f" % Vy)
            return Sx
        else:
            return 0.0001 * Sx

    def deriv(self,z, t):
        I = self.m1 * self.L1 ** 2 + self.m2 * self.L2 ** 2
        a = self.g * (self.m1 * self.L1 - self.m2 * self.L2)

        # z[0] = theta
        # z[1] = theta_dot
        dz0 = z[1]
        dz1 = a * np.sin(z[0]) / I
        return np.array([dz0, dz1])

    def runTreb(self, m1 = 30, m2 = 1, L2 = 1, L1 = 1.0, endtime = 3.0, dt = 0.01, zinit = np.array([45 * np.pi / 180, 0])):
        self.m1 = m1
        self.m2 = m2
        self.L2 = L2
        self.L1 = L1
        self.endtime = endtime
        self.dt = dt
        self.zinit = zinit
        self.h = -np.cos(45 * np.pi / 180) * self.L2
        self.t = np.linspace(0, self.endtime, endtime/dt)

        z = odeint(self.deriv, self.zinit, self.t)

        self.x1 = self.L1 * np.sin(z[:, 0])
        self.y1 = self.L1 * np.cos(z[:, 0])
        self.x2 = -self.L2 * np.sin(z[:, 0])
        self.y2 = -self.L2 * np.cos(z[:, 0])

        Sx = [self.ThrowRange(z[i, 0], z[i, 1]) for i in range(len(self.t))]

        self.Sx = np.asarray(Sx)
        self.maxSx = np.max(Sx)
        self.rangeThrown = self.maxSx
        # Max throw range throw angle.
        self.SxMaxThrowAngle = self.throwAngle(z[np.nanargmax(Sx),0])

        return self.rangeThrown, self.SxMaxThrowAngle

if __name__ == "__main__":
    zinit = np.array([45 * np.pi / 180, 0])
    testTreb = pendulumTreb("test")
    testTreb.runTreb(m1 = 40, zinit = zinit)
    print(testTreb.rangeThrown)
    print(np.rad2deg(testTreb.SxMaxThrowAngle))