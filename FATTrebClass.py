# Class for the Floating Arm Trebuchet
# Created by Graham Bell 08/07/2016

import numpy as np
from numpy import sin, cos, pi
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import trebuchetTools

class FATTreb:
    def __init__(self,name='treb1', beamDensity = 1.0, g=9.81):
        self.name = name
        self.beamDensity = beamDensity
        self.g = g

    def throwRange(self, xPos, yPos, alpha):
        # Just take the numerical gradient of the x and y positions for easy methods to get velocities.
        self.x3Dot = np.gradient(xPos, edge_order=2)/self.dt
        self.y3Dot = np.gradient(yPos, edge_order=2)/self.dt
        Sx = self.x3Dot * ((self.y3Dot + np.sqrt((self.y3Dot ** 2) + (2 * self.g * (yPos - self.h)))) / self.g) + xPos
        Sx[self.x3Dot < 0] = np.nan
        Sx[self.y3Dot < 0] = np.nan
        Sx[alpha > 2*np.pi] = np.nan
        return Sx

    def deriv(self, z, t):
        # z[0] = theta
        # z[1] = alpha
        # z[2] = theta_dot
        # z[3] = alpha_dot
        L1 = self.L1
        L2 = self.L2
        L3 = self.L3
        m1 = self.m1
        m3 = self.m3
        m_b = self.m_b
        g = self.g
        dz0 = z[2]
        dz1 = z[3]
        dz2 = 3 * (-4 * L1 ** 2 * m1 * sin(2 * z[0]) * z[2] ** 2 + 4 * L1 ** 2 * m3 * sin(2 * z[0]) * z[
            2] ** 2 + L1 ** 2 * m_b * sin(2 * z[0]) * z[2] ** 2 + 8 * L1 * L2 * m3 * sin(2 * z[0]) * z[
                       2] ** 2 + 2 * L1 * L2 * m_b * sin(2 * z[0]) * z[2] ** 2 - 4 * L1 * L3 * m3 * cos(z[1] - z[0]) *
                   z[3] ** 2 - 4 * L1 * L3 * m3 * cos(z[1] + z[0]) * z[3] ** 2 + 8 * L1 * g * m1 * sin(
            z[0]) + 4 * L1 * g * m_b * sin(z[0]) + L2 ** 2 * m_b * sin(2 * z[0]) * z[2] ** 2 - 8 * L2 * L3 * m3 * cos(
            z[1] - z[0]) * z[3] ** 2 - 8 * L2 * g * m3 * sin(z[0]) - 4 * L2 * g * m_b * sin(z[0]) - 8 * m3 * (
                   L1 * sin(z[1]) * cos(z[0]) + L2 * sin(z[1] - z[0])) * (
                   L1 * sin(z[1]) * sin(z[0]) * z[2] ** 2 + L2 * cos(z[1] - z[0]) * z[2] ** 2 + g * cos(z[1]))) / (
              -12 * L1 ** 2 * m1 * cos(2 * z[0]) + 12 * L1 ** 2 * m1 + 12 * L1 ** 2 * m3 * cos(
                  2 * z[0]) + 12 * L1 ** 2 * m3 + 3 * L1 ** 2 * m_b * cos(
                  2 * z[0]) + 11 * L1 ** 2 * m_b + 24 * L1 * L2 * m3 * cos(
                  2 * z[0]) + 24 * L1 * L2 * m3 + 6 * L1 * L2 * m_b * cos(
                  2 * z[0]) - 2 * L1 * L2 * m_b + 24 * L2 ** 2 * m3 + 3 * L2 ** 2 * m_b * cos(
                  2 * z[0]) + 11 * L2 ** 2 * m_b - 24 * m3 * (L1 * sin(z[1]) * cos(z[0]) + L2 * sin(z[1] - z[0])) ** 2)
        dz3 = (-3 * (L1 * sin(z[1]) * cos(z[0]) + L2 * sin(z[1] - z[0])) * (
        -4 * L1 ** 2 * m1 * sin(2 * z[0]) * z[2] ** 2 + 4 * L1 ** 2 * m3 * sin(2 * z[0]) * z[
            2] ** 2 + L1 ** 2 * m_b * sin(2 * z[0]) * z[2] ** 2 + 8 * L1 * L2 * m3 * sin(2 * z[0]) * z[
            2] ** 2 + 2 * L1 * L2 * m_b * sin(2 * z[0]) * z[2] ** 2 - 4 * L1 * L3 * m3 * cos(z[1] - z[0]) * z[
            3] ** 2 - 4 * L1 * L3 * m3 * cos(z[1] + z[0]) * z[3] ** 2 + 8 * L1 * g * m1 * sin(
            z[0]) + 4 * L1 * g * m_b * sin(z[0]) + L2 ** 2 * m_b * sin(2 * z[0]) * z[2] ** 2 - 8 * L2 * L3 * m3 * cos(
            z[1] - z[0]) * z[3] ** 2 - 8 * L2 * g * m3 * sin(z[0]) - 4 * L2 * g * m_b * sin(z[0])) + (
               L1 * sin(z[1]) * sin(z[0]) * z[2] ** 2 + L2 * cos(z[1] - z[0]) * z[2] ** 2 + g * cos(z[1])) * (
               -24 * L1 ** 2 * m1 * cos(z[0]) ** 2 + 24 * L1 ** 2 * m1 + 24 * L1 ** 2 * m3 * cos(
                   z[0]) ** 2 + 6 * L1 ** 2 * m_b * cos(z[0]) ** 2 + 8 * L1 ** 2 * m_b + 48 * L1 * L2 * m3 * cos(
                   z[0]) ** 2 + 12 * L1 * L2 * m_b * cos(
                   z[0]) ** 2 - 8 * L1 * L2 * m_b + 24 * L2 ** 2 * m3 + 6 * L2 ** 2 * m_b * cos(
                   z[0]) ** 2 + 8 * L2 ** 2 * m_b)) / (L3 * (
        -12 * L1 ** 2 * m1 * cos(2 * z[0]) + 12 * L1 ** 2 * m1 + 12 * L1 ** 2 * m3 * cos(
            2 * z[0]) + 12 * L1 ** 2 * m3 + 3 * L1 ** 2 * m_b * cos(
            2 * z[0]) + 11 * L1 ** 2 * m_b + 24 * L1 * L2 * m3 * cos(
            2 * z[0]) + 24 * L1 * L2 * m3 + 6 * L1 * L2 * m_b * cos(
            2 * z[0]) - 2 * L1 * L2 * m_b + 24 * L2 ** 2 * m3 + 3 * L2 ** 2 * m_b * cos(
            2 * z[0]) + 11 * L2 ** 2 * m_b - 24 * m3 * (L1 * sin(z[1]) * cos(z[0]) + L2 * sin(z[1] - z[0])) ** 2))
        return np.array([dz0, dz1, dz2, dz3])

    def runTreb(self, L1=1.0, L2=1.0, L3=1.0, m1=50, m3=1.0, endtime=2.0, dt=0.01,
                zinit=np.array([45.0*pi/180, 1.0*pi/180, 0.0, 0.0])):
        self.m1 = m1
        self.m3 = m3
        self.L1 = L1
        self.L3 = L3
        self.L2 = L2
        self.zinit = zinit
        #zinit = np.array([45 * np.pi / 180, 0])

        # Set the beam mass to almost zero for the test optimisation case.
        self.m_b = (self.L1 + self.L2) * self.beamDensity

        self.endtime = endtime
        self.dt = dt
        self.t = np.linspace(0, self.endtime, endtime/dt)
        #self.zinit = np.array([45.0 * pi / 180, 1.0 * pi / 180, 0.0, 0.0])
        self.z = odeint(self.deriv, zinit, self.t)

        self.x4 = -self.L1 * np.sin(self.z[:, 0])
        self.y1 = self.L1 * np.cos(self.z[:, 0])
        self.h = -np.cos(45.0 * pi / 180) * self.L2
        self.xb = self.x4 - ((self.L2 - self.L1) / 2) * np.sin(self.z[:, 0])
        self.yb = -((self.L2 - self.L1) / 2) * np.cos(self.z[:, 0])

        self.x2 = self.x4 - self.L2 * np.sin(self.z[:, 0])
        self.y2 = -self.L2 * np.cos(self.z[:, 0])

        self.x3 = self.x2 + self.L3 * np.cos(self.z[:, 1])
        self.y3 = self.y2 - self.L3 * np.sin(self.z[:, 1])

        # Defined but used only for animation
        self.x0 = self.x4 - ((self.L2 - self.L1) / 2) * np.sin(self.z[:, 0])
        self.y0 = -((self.L2 - self.L1) / 2) * np.cos(self.z[:, 0])

        # Grab the largest range value. ThrowRange function args: theta,thetadot,alpha,alphadot
        # Grab the largest range value.
        self.Sx, self.x3Dot, self.y3Dot, self.maxThrowIndex, self.maxThrowRange, self.SxMaxThrowAngle = \
            trebuchetTools.throwRange(xPos=self.x3, yPos=self.y3, dt=self.dt, alpha=self.z[:, 1], g=self.g, h=self.h,
                                 validateThrow=True, returnMaxes=True)

        return self.maxThrowRange, self.SxMaxThrowAngle

    def totalKineticEnergy(self):
        TKE = self.L1**2*self.m1*np.sin(self.z[:, 0])**2*self.z[:, 2]**2/2 + self.m3*((self.L2*np.sin(self.z[:, 0])*self.z[:, 2] - self.L3*np.cos(self.z[:, 1])*self.z[:, 3])**2 + (-self.L1*np.cos(self.z[:, 0])*self.z[:, 2] - self.L2*np.cos(self.z[:, 0])*self.z[:, 2] - self.L3*np.sin(self.z[:, 1])*self.z[:, 3])**2)/2 + self.m_b*(-self.L1*np.cos(self.z[:, 0])*self.z[:, 2] - (-self.L1/2 + self.L2/2)*np.cos(self.z[:, 0])*self.z[:, 2])**2/2 + self.m_b*(self.L1**2 - self.L1*self.L2 + self.L2**2)*self.z[:, 2]**2/6
        return TKE

    def totalPotentialEnergy(self):
        CWRefZeroHeight = self.L1
        TPE = self.m3 * self.g * self.y3 + self.m1 * self.g * (self.y1+CWRefZeroHeight) + self.m_b * self.g * self.yb
        return TPE

    def init(self):
        return []

    def animate(self, i):
        patches = []

        # Vertical and horizontal lines:
        # Horizontal rail
        patches.append(ax.add_line(plt.Line2D((-1, 1), (0, 0), lw=1.7, alpha=0.3, color='k', linestyle='--')))
        # Vertical Rail
        patches.append(ax.add_line(plt.Line2D((0, 0), (-1.5, 1.5), lw=1.7, alpha=0.3, color='k', linestyle='--')))

        # x4,y4 - x1,y1 line

        patches.append(ax.add_line(plt.Line2D((self.x4[i % len(self.t)], 0), (0, self.y1[i % len(self.t)]), lw=2.5)))
        # x4,y4-x2,y2 line
        patches.append(ax.add_line(plt.Line2D((self.x2[i % len(self.t)], self.x4[i % len(self.t)]),
                                              (self.y2[i % len(self.t)], 0), lw=2.5)))
        # x2,y2-x3,y3 line
        patches.append(ax.add_line(plt.Line2D((self.x2[i % len(self.t)], self.x3[i % len(self.t)]),
                                              (self.y2[i % len(self.t)], self.y3[i % len(self.t)]), lw=1.8)))

        # m1 CW circle
        patches.append(ax.add_patch(plt.Circle((0, self.y1[i % len(self.t)]), 0.2, color='k')))
        # Beam axle circle
        patches.append(ax.add_patch(plt.Circle((self.x4[i % len(self.t)], 0), 0.08, color='r')))
        # beam centre circle
        patches.append(ax.add_patch(plt.Circle((self.x0[i % len(self.t)], self.y0[i % len(self.t)]), 0.05, color='k')))
        # Projectile circle
        patches.append(ax.add_patch(plt.Circle((self.x3[i % len(self.t)], self.y3[i % len(self.t)]), 0.1, color='r')))
        #patches.append(ax.text(-2, 2, "Total Energy: %.3f" % totalEnergy[i]))
        return patches

    def plotTreb(self, ax, i, lineThickness=0.4, railThickness=0.3, CWRadius=0.2, projRadius=0.5, beamCentreRadius=0.2,
                 imageAlpha=1.0):
        # Vertical and horizontal lines:
        # Horizontal rail
        #plt.clf()
        #fig = plt.figure(figsize=(10, 6), dpi=80)

        # fig = plt.figure()
        #
        linewidths = 0.8
        ax.add_line(plt.Line2D((-1, 1), (0, 0), lw=railThickness, alpha=0.3*imageAlpha, color='k', linestyle='--',))
        # Vertical Rail
        ax.add_line(plt.Line2D((0, 0), (-1.5, 1.5), lw=railThickness, alpha=0.3*imageAlpha, color='k', linestyle='--'))

        # x4,y4 - x1,y1 line

        ax.add_line(plt.Line2D((self.x4[i % len(self.t)], 0), (0, self.y1[i % len(self.t)]), alpha=imageAlpha,
                               lw=lineThickness, color='k'))
        # x4,y4-x2,y2 line
        ax.add_line(plt.Line2D((self.x2[i % len(self.t)], self.x4[i % len(self.t)]),
                                              (self.y2[i % len(self.t)], 0), lw=lineThickness, alpha=imageAlpha,
                               color='k'))
        # x2,y2-x3,y3 line
        ax.add_line(plt.Line2D((self.x2[i % len(self.t)], self.x3[i % len(self.t)]),
                                              (self.y2[i % len(self.t)], self.y3[i % len(self.t)]), lw=lineThickness,
                               alpha=imageAlpha, zorder=2, color='k'))

        # m1 CW circle
        ax.add_patch(plt.Circle((0, self.y1[i % len(self.t)]), CWRadius, color='#e0e4cc', alpha=imageAlpha, zorder=3))
        # Beam axle circle
        ax.add_patch(plt.Circle((self.x4[i % len(self.t)], 0), 0.1, color='#69d0e7', alpha=0.5*imageAlpha, zorder=3))
        # beam centre circle
        ax.add_patch(plt.Circle((self.x0[i % len(self.t)], self.y0[i % len(self.t)]), beamCentreRadius, color='k',
                                alpha=0.5*imageAlpha, zorder=3))
        # Projectile circle
        ax.add_patch(plt.Circle((self.x3[i % len(self.t)], self.y3[i % len(self.t)]), projRadius, color='#f38630',
                                alpha=imageAlpha, zorder=3))
        return ax

    def optimiserThrowTreb(self, x, returnThrowAngle=False):
        [L2, L3] = x

        throwRange, throwAngle = self.runTreb(m1=self.m1, m3=self.m3, L2=L2, L3=L3, endtime=self.endtime, dt=self.dt, zinit=self.zinit)
        # print("Optimizer Trying:", 'L2:', L2, 'L3:', L3, 'Range:',throwRange, 'throw angle:', np.rad2deg(throwAngle))
        if np.isnan(throwRange):
            throwRange = 0.01
        if returnThrowAngle is True:
            return -throwRange, throwAngle
        else:
            return -throwRange

if __name__ == "__main__":

    treb = FATTreb("This Treb", beamDensity=1.0)
    zinit = np.array([45.0 * np.pi / 180, 0.1 * np.pi / 180, 0.0, 0.0])
    range, throwAngle = treb.runTreb(L2=3.0, L3=3.0,m1=100,endtime=2.0, zinit=zinit)
    print('range:',range, 'Throw angle:', np.rad2deg(throwAngle))
    #print(testFAT.runFATTreb(L2 = 2.36, L3 = 2.09, m1=51, endtime=3.5, zinit=zinit))
    import matplotlib.pyplot as plt
    from matplotlib import animation

    CWEnergy = 9.81 * 100 * (1.0 + np.sin(np.pi / 4))
    print(CWEnergy)
    if True:
        TKE = treb.totalKineticEnergy()
        TPE = treb.totalPotentialEnergy()
        projPE = trebuchetTools.projPE(mass=treb.m3, yPos=treb.y3, g=treb.g, refHeight=treb.h)
        projKE = trebuchetTools.projKE(mass=treb.m3, xDot=treb.x3Dot, yDot=treb.y3Dot)
        plt.plot(treb.t, TKE, label='TKE')
        plt.plot(treb.t, TPE, label='TPE')
        plt.plot(treb.t, projKE, label='projKE')
        plt.plot(treb.t, projPE, label='projPE')
        plt.axhline(y=CWEnergy)
        plt.axvline(x=treb.t[treb.maxThrowIndex])
        plt.legend()
        plt.show()
        CWEnergy = 9.81*100*(1.0+np.sin(np.pi/4))
        print(CWEnergy)

    fig = plt.figure(figsize=(10, 6), dpi=80)
    plt.subplot(1, 1, 1, aspect='equal')

    xlimits = plt.xlim(-5, 5)
    ylimits = plt.ylim(-5, 5)
    ax = plt.axes(xlim=xlimits, ylim=ylimits, aspect='equal')

    fps = 24
    anim = animation.FuncAnimation(fig, treb.animate, init_func=treb.init, frames=len(treb.t), interval=1,
                                   blit=True)
    plt.show()
