# Created by Graham Bell on 08/07/2016
# Class model for the standard trebuchet

import numpy as np
from numpy import sin, cos, pi
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import trebuchetTools

class standardTreb:
    def __init__(self,name='treb1', beamDensity = 1.0,g=9.81):
        self.name = name
        self.beamDensity = beamDensity
        self.g = g

    def deriv(self, z, t):
        L1 = self.L1
        L2 = self.L2
        L3 = self.L3
        L4 = self.L4
        g = self.g
        m1 = self.m1
        m3 = self.m3
        m_b = self.m_b

        # z[0] = theta
        # z[1] = phi
        # z[2] = alpha
        # z[3] = theta_dot
        # z[4] = phi_dot
        # z[5] = alpha_dot

        dz0 = z[3]
        dz1 = z[4]
        dz2 = z[5]
        dz3 = 3 * (L1 ** 2 * m1 * sin(2 * z[1] + 2 * z[0]) * z[3] ** 2 - 2 * L1 * L4 * m1 * sin(z[1] + z[0]) * z[
            4] ** 2 - L1 * g * m1 * sin(2 * z[1] + z[0]) - L1 * g * m1 * sin(z[0]) - L1 * g * m_b * sin(
            z[0]) + L2 ** 2 * m3 * sin(2 * z[2] - 2 * z[0]) * z[3] ** 2 + 2 * L2 * L3 * m3 * cos(z[2] - z[0]) * z[
                       5] ** 2 + L2 * g * m3 * sin(2 * z[2] - z[0]) + L2 * g * m3 * sin(z[0]) + L2 * g * m_b * sin(
            z[0])) / (2 * (3 * L1 ** 2 * m1 * cos(
            z[1] + z[0]) ** 2 - 3 * L1 ** 2 * m1 - L1 ** 2 * m_b + L1 * L2 * m_b + 3 * L2 ** 2 * m3 * sin(
            z[2] - z[0]) ** 2 - 3 * L2 ** 2 * m3 - L2 ** 2 * m_b))
        dz4 = (-3 * L1 * L2 * m3 * (L2 * cos(z[2] - z[0]) * z[3] ** 2 + g * cos(z[2])) * sin(z[2] - z[0]) * cos(
            z[1] + z[0]) + 3 * L1 * (
               2 * L1 * L4 * m1 * sin(z[1] + z[0]) * z[4] ** 2 + 2 * L1 * g * m1 * sin(z[0]) + L1 * g * m_b * sin(
                   z[0]) - 2 * L2 * L3 * m3 * cos(z[2] - z[0]) * z[5] ** 2 - 2 * L2 * g * m3 * sin(
                   z[0]) - L2 * g * m_b * sin(z[0])) * cos(z[1] + z[0]) / 2 + (
               L1 * sin(z[1] + z[0]) * z[3] ** 2 - g * sin(z[1])) * (
               -3 * L1 ** 2 * m1 - L1 ** 2 * m_b + L1 * L2 * m_b + 3 * L2 ** 2 * m3 * sin(
                   z[2] - z[0]) ** 2 - 3 * L2 ** 2 * m3 - L2 ** 2 * m_b)) / (L4 * (3 * L1 ** 2 * m1 * cos(
            z[1] + z[0]) ** 2 - 3 * L1 ** 2 * m1 - L1 ** 2 * m_b + L1 * L2 * m_b + 3 * L2 ** 2 * m3 * sin(
            z[2] - z[0]) ** 2 - 3 * L2 ** 2 * m3 - L2 ** 2 * m_b))
        dz5 = (-3 * L1 * L2 * m1 * (L1 * sin(z[1] + z[0]) * z[3] ** 2 - g * sin(z[1])) * sin(z[2] - z[0]) * cos(
            z[1] + z[0]) + 3 * L2 * (
               2 * L1 * L4 * m1 * sin(z[1] + z[0]) * z[4] ** 2 + 2 * L1 * g * m1 * sin(z[0]) + L1 * g * m_b * sin(
                   z[0]) - 2 * L2 * L3 * m3 * cos(z[2] - z[0]) * z[5] ** 2 - 2 * L2 * g * m3 * sin(
                   z[0]) - L2 * g * m_b * sin(z[0])) * sin(z[2] - z[0]) / 2 + (
               L2 * cos(z[2] - z[0]) * z[3] ** 2 + g * cos(z[2])) * (3 * L1 ** 2 * m1 * cos(z[1] + z[
            0]) ** 2 - 3 * L1 ** 2 * m1 - L1 ** 2 * m_b + L1 * L2 * m_b - 3 * L2 ** 2 * m3 - L2 ** 2 * m_b)) / (L3 * (
        3 * L1 ** 2 * m1 * cos(
            z[1] + z[0]) ** 2 - 3 * L1 ** 2 * m1 - L1 ** 2 * m_b + L1 * L2 * m_b + 3 * L2 ** 2 * m3 * sin(
            z[2] - z[0]) ** 2 - 3 * L2 ** 2 * m3 - L2 ** 2 * m_b))
        return np.array([dz0, dz1, dz2, dz3, dz4, dz5])

    def runTreb(self, m1=40.0, m3 =1.0, L1=1.0, L2=2.0, L3=2.0, dt=0.01, endtime=2.0,
                zinit=np.array([45*pi/180, 1*pi/180, 1*pi/180, 0, 0, 0])):
        self.m1 = m1
        self.m3 = m3
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.m_b = (self.L1 + self.L2) * self.beamDensity
        self.zinit = zinit
        self.h = -np.cos(self.zinit[0]) * self.L2
        self.L4 = - self.L1 - self.h
        if self.L4 < 0.0:
            # If L4 is negative it gives the CW extra height.
            # Set it esentially to 0
            self.L4 = 0.05
        #print('L4:',self.L4)

        self.endtime = endtime
        self.dt = dt
        self.t = np.linspace(0, endtime, endtime/dt)
        self.zinit = np.array([45 * pi / 180, 1 * pi / 180, 1 * pi / 180, 0, 0, 0])
        self.z = odeint(self.deriv, self.zinit, self.t)

        self.x1 = self.L1 * np.sin(self.z[:, 0])
        self.y1 = self.L1 * np.cos(self.z[:, 0])
        self.x2 = -self.L2 * np.sin(self.z[:, 0])
        self.y2 = -self.L2 * np.cos(self.z[:, 0])

        self.x3 = self.x2 + self.L3 * np.cos(self.z[:, 2])
        self.y3 = self.y2 - self.L3 * np.sin(self.z[:, 2])

        self.x4 = self.x1 + self.L4 * np.sin(self.z[:, 1])
        self.y4 = self.y1 - self.L4 * np.cos(self.z[:, 1])
        self.yb = -((self.L2 - self.L1) / 2) * np.cos(self.z[:, 0])

        # Grab the largest range value.
        self.Sx, self.x3Dot, self.y3Dot, self.maxThrowIndex, self.maxThrowRange, self.SxMaxThrowAngle = \
            trebuchetTools.throwRange(xPos=self.x3, yPos=self.y3, dt=self.dt, alpha=self.z[:, 2], g=self.g, h=self.h,
                                 validateThrow=True, returnMaxes=True)
        return self.maxThrowRange, self.SxMaxThrowAngle

    def totalKineticEnergy(self):
        TKE = self.m1*((-self.L1*np.sin(self.z[:, 0])*self.z[:, 3] + self.L4*np.sin(self.z[:, 1])*self.z[:, 4])**2 + (self.L1*np.cos(self.z[:, 0])*self.z[:, 3] + self.L4*np.cos(self.z[:, 1])*self.z[:, 4])**2)/2 + self.m3*((self.L2*np.sin(self.z[:, 0])*self.z[:, 3] - self.L3*np.cos(self.z[:, 2])*self.z[:, 5])**2 + (-self.L2*np.cos(self.z[:, 0])*self.z[:, 3] - self.L3*np.sin(self.z[:, 2])*self.z[:, 5])**2)/2 + self.m_b*(self.L1**2 - self.L1*self.L2 + self.L2**2)*self.z[:, 3]**2/6
        return TKE

    def totalPotentialEnergy(self):
        CWRefZeroHeight = self.L1 + self.L4
        TPE = self.g*self.m1*(CWRefZeroHeight + self.L1*np.cos(self.z[:, 0]) - self.L4*np.cos(self.z[:, 1])) + self.g*self.m3*(-self.L2*np.cos(self.z[:, 0]) - self.L3*np.sin(self.z[:, 2])) + self.g*self.m_b*(self.L1/2 - self.L2/2)*np.cos(self.z[:, 0])
        return TPE

    def init(self):
        return []

    def animate(self, i):
        patches = []
        trebarmwidth = 1.0
        # L1 line
        patches.append(ax.add_line(plt.Line2D((0, self.x1[i % len(self.t)]), (0, self.y1[i % len(self.t)]), lw=trebarmwidth)))

        # L2 line
        patches.append(ax.add_line(plt.Line2D((0, self.x2[i % len(self.t)]), (0, self.y2[i % len(self.t)]), lw=trebarmwidth)))
        # L4 line
        patches.append(ax.add_line(plt.Line2D((self.x1[i % len(self.t)], self.x4[i % len(self.t)]),
                                              (self.y1[i % len(self.t)], self.y4[i % len(self.t)]), lw=trebarmwidth)))
        # L3 line
        patches.append(ax.add_line(plt.Line2D((self.x2[i % len(self.t)], self.x3[i % len(self.t)]),
                                              (self.y2[i % len(self.t)], self.y3[i % len(self.t)]),
                                              lw=trebarmwidth, color='c')))

        # x1 hinge position circle
        # patches.append(ax.add_patch(plt.Circle((x1[i % len(t)],y1[i % len(t)]),0.1,color='b') ))
        # CW circle
        patches.append(ax.add_patch(plt.Circle((self.x4[i % len(self.t)], self.y4[i % len(self.t)]), 0.2, color='k')))
        # Beam tip circle
        # patches.append(ax.add_patch(plt.Circle((x2[i % len(t)],y2[i % len(t)]),0.1,color='r') ))
        # Projectile tip circle
        patches.append(ax.add_patch(plt.Circle((self.x3[i % len(self.t)], self.y3[i % len(self.t)]), 0.1, color='r')))
        # Center dot
        patches.append(ax.add_patch(plt.Circle((0, 0), 0.05, color='k')))
        return patches

    def plotTreb(self, ax, i, lineThickness=0.4, railThickness=0.3, CWRadius=0.2, projRadius=0.5, beamCentreRadius=0.2,
                 imageAlpha=1.0):
        # L1 line
        ax.add_line(plt.Line2D((0, self.x1[i % len(self.t)]), (0, self.y1[i % len(self.t)]), lw=lineThickness,
                               alpha=imageAlpha, color='k'))

        # L2 line
        ax.add_line(plt.Line2D((0, self.x2[i % len(self.t)]), (0, self.y2[i % len(self.t)]), lw=lineThickness,
                               alpha=imageAlpha, color='k'))
        # L4 line
        ax.add_line(plt.Line2D((self.x1[i % len(self.t)], self.x4[i % len(self.t)]),
                                              (self.y1[i % len(self.t)], self.y4[i % len(self.t)]), lw=lineThickness,
                               color='k', alpha=imageAlpha))
        # L3 line
        ax.add_line(plt.Line2D((self.x2[i % len(self.t)], self.x3[i % len(self.t)]),
                                              (self.y2[i % len(self.t)], self.y3[i % len(self.t)]),
                                              lw=lineThickness, color='k', alpha=imageAlpha))

        # x1 hinge position circle
        # patches.append(ax.add_patch(plt.Circle((x1[i % len(t)],y1[i % len(t)]),0.1,color='b') ))
        # CW circle
        ax.add_patch(plt.Circle((self.x4[i % len(self.t)], self.y4[i % len(self.t)]), CWRadius, color='#e0e4cc',
                                alpha=imageAlpha, zorder=3))
        # Beam tip circle
        # patches.append(ax.add_patch(plt.Circle((x2[i % len(t)],y2[i % len(t)]),0.1,color='r') ))
        # Projectile tip circle
        ax.add_patch(plt.Circle((self.x3[i % len(self.t)], self.y3[i % len(self.t)]), projRadius, color='#f38630',
                                zorder=3, alpha=imageAlpha))
        # Center dot
        ax.add_patch(plt.Circle((0, 0), 0.1, color='k', alpha=imageAlpha, zorder=3))
        return ax

    def optimiserThrowTreb(self, x, returnThrowAngle=False):
        [L2, L3] = x

        throwRange, throwAngle = self.runTreb(m1=self.m1, m3=self.m3, L2=L2, L3=L3, endtime=self.endtime, dt=self.dt,
                                              zinit=self.zinit)

        if np.isnan(throwRange):
            throwRange = 0.1
        print("Optimizer Trying:", 'L2:', L2, 'L3:', L3, 'Range:',throwRange, 'throw angle:', np.rad2deg(throwAngle))

        if returnThrowAngle is True:
            return -throwRange, throwAngle
        else:
            return -throwRange

if __name__ == "__main__":
    treb = standardTreb(name="TestTreb")
    zinit = np.array([45 * pi / 180, 0.0 * pi / 180, 0.0 * pi / 180, 0, 0, 0])
    print('Standard Trebuchet:')
    range, throwAngle = treb.runTreb(m1=100.0, L2=3.0, L3=3.0, zinit=zinit, endtime=2.0, dt=0.005)
    print('Throw range:', range, 'throwAngle:', np.rad2deg(throwAngle))

    # Animate
    if True:
        CWEnergy = 9.81 * 100 * (1.0 + np.sin(np.pi / 4))
        print(CWEnergy)
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

    import matplotlib.pyplot as plt
    from matplotlib import animation


    fig = plt.figure(figsize=(10, 6), dpi=80)
    plt.subplot(1, 1, 1, aspect='equal')

    xlimits = plt.xlim(-5, 5)
    ylimits = plt.ylim(-5, 5)
    ax = plt.axes(xlim=xlimits, ylim=ylimits, aspect='equal')

    fps = 24
    anim = animation.FuncAnimation(fig, treb.animate, init_func=treb.init,
                                   frames=len(treb.t), interval=1, blit=True)
    plt.show()