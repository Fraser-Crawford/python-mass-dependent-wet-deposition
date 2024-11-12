from dataclasses import dataclass

import numpy as np

from distribution import Distribution
import scipy
import numpy.typing as npt

@dataclass
class LogNormal(Distribution):
    mean_diameter: float
    log_std: float
    def get_number_concentration(self, diameter: float) -> float:
        return self.number/(np.sqrt(2*np.pi)*self.log_std*diameter)*np.exp(-(np.log10(diameter)-np.log10(self.mean_diameter))**2/(2*self.log_std**2))

    def integrate_number_concentration(self, lower_diameters: npt.NDArray[np.float_], upper_diameters: npt.NDArray[np.float_],) -> npt.NDArray[np.float_]:
        return np.array([scipy.integrate.quad(self.get_number_concentration, lower_diameter, upper_diameter)[0] for lower_diameter, upper_diameter in zip(lower_diameters, upper_diameters)])