from typing import Callable

import numpy as np
from dataclasses import dataclass
import scipy.constants as const
from scipy.integrate import quad

from marshall_palmer import MarshallPalmer
import numpy.typing as npt
AIR_DENSITY = 1.292
WATER_DENSITY = 1000
GRAVITY = 9.81
T = 293
MU_AIR = 1.72e-5
MU_WATER = 1.787e-3
MEAN_FREE_PATH = 6.19e-8
OMEGA = MU_WATER / MU_AIR


@dataclass
class Droplet:
    diameter: float
    density: float

    @property
    def reynolds(self):
        cdre2 = 4.0 * self.diameter**3 * AIR_DENSITY * self.density * GRAVITY * self.slip / (3.0 * MU_AIR**2)
        a = 0.095202
        b = 1.1058
        c = 1.4338
        return np.where(cdre2 < 2.4,
                            cdre2 / 24.0,
                        np.where(cdre2 < 1.1e5,
                            np.power(10.0, (-b + np.sqrt(b**2 - 4.0 * a * c + 4.0 * a * np.log10(cdre2))) / (2.0 * a)),
                            np.sqrt(cdre2 / 0.44)))

    @property
    def slip(self):
        return 1 + 2 * MEAN_FREE_PATH / self.diameter * (
                    1.257 + 0.4 * np.exp(-1.1 * self.diameter / (2.0 * MEAN_FREE_PATH)) * MEAN_FREE_PATH)

    @property
    def terminal_velocity(self):
        return MU_AIR * self.reynolds / (AIR_DENSITY * self.diameter)

    @property
    def diffusion(self):
        return const.Boltzmann * T * self.slip / (3.0 * np.pi * MU_AIR * self.diameter)

    @property
    def schmidt(self):
        return MU_AIR / (AIR_DENSITY * self.diffusion)

    @property
    def tau(self):
        return (self.density - AIR_DENSITY) * self.diameter**2 * self.slip / (18 * MU_AIR)

    def stokes(self,tau):
        return 2 * tau * (self.terminal_velocity - tau*GRAVITY) / self.diameter

    @property
    def critical_stokes(self):
        return (1.2 + np.log(1.0 + self.reynolds) / 12.0) / (1 + np.log(1 + self.reynolds))


def slinn(aerosol: Droplet, droplet: Droplet) -> float:
    phi = aerosol.diameter / droplet.diameter
    schmidt = aerosol.schmidt
    reynolds = droplet.reynolds
    diffusion = 4 * (1 + 0.4 * np.sqrt(reynolds) * np.cbrt(schmidt) + 0.16 * np.sqrt(reynolds * schmidt)) / (
                reynolds * schmidt)
    interception = 4 * phi * (1 / OMEGA + phi * (1 + 2 * np.sqrt(reynolds)))
    stokes = droplet.stokes(aerosol.tau)
    critical_stokes = droplet.critical_stokes
    if stokes >= critical_stokes:
        impaction = np.power((stokes - critical_stokes) / (stokes - critical_stokes + 2 / 3), 3.0 / 2.0)
    else:
        impaction = 0
    return diffusion + interception + impaction

@dataclass
class Scavenging:
    aerosol: Droplet
    collection: Callable[[Droplet,Droplet],float]
    precipitation: MarshallPalmer
    def integral(self, diameter:float)->float:
        droplet = Droplet(diameter,WATER_DENSITY)
        volume_swept = np.pi/4*(self.aerosol.diameter+diameter)**2*(droplet.terminal_velocity-self.aerosol.terminal_velocity)
        return volume_swept*self.collection(self.aerosol,droplet)*self.precipitation.get_number_concentration(diameter)

def scavenging(aerosol_diameters: npt.NDArray[np.float_], aerosol_density: float, precipitation: MarshallPalmer,
               collection: Callable[[Droplet, Droplet], float]) -> npt.NDArray[np.float_]:
    return np.array([quad(Scavenging(Droplet(aerosol_diameter,aerosol_density),collection,precipitation).integral,1e-4,1e-1)[0] for aerosol_diameter in aerosol_diameters])
